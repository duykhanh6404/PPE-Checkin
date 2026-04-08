import cv2
import time
import os
import threading
from models.yolo_model import get_yolo_model
from models.face_model import app_face
from database.database import SessionLocal, DetectionLog, Employee
import datetime
import json
import numpy as np
from numpy.linalg import norm

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)

def get_intersection_ratio(boxA, boxB):
    # Phần trăm diện tích boxB chui tọt vào bên trong boxA
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxBArea = max(1, (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1))
    
    return interArea / float(boxBArea)

class CameraStream:
    """Class xử lý stream camera, detect YOLO realtime và log lịch sử"""
    def __init__(self, source=0):
        # source có thể là 0 (webcam) hoặc đường dẫn RTSP (rtsp://...)
        self.source = source
        self.cap = None
        self.is_running = False
        self.model = get_yolo_model()
        
        # Thêm biến xử lý chống lag IP Camera (như DroidCam) bằng Thread riêng
        self.latest_frame = None
        self.read_thread = None
        
        self.employee_db = self._load_employee_db()
        self.frame_counter = 0
        self.stranger_cooldown = 0
        self.tracked_identities = [] # Dành cho Simple Tracker lưu Identity qua các frame
        
        # Lock để đảm bảo an toàn nếu nhiều người cùng truy cập xem ảnh
        self.lock = threading.Lock()
        
        # Thư mục chứa ảnh bị lỗi PPE (đảm bảo luôn nằm trong ppe-system)
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.snapshot_dir = os.path.join(BASE_DIR, "static", "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Tránh spam database, ít nhất 5 giây mới lưu lỗi 1 lần cho mỗi trường hợp hiển thị
        self.last_log_time = 0
        
        # Lưu trạng thái mới nhất cho giao diện "KIỂM TRA" fetch
        self.latest_status = {
            "Áo bảo hộ": "Chưa rõ",
            "Mũ bảo hộ": "Chưa rõ",
            "Giày bảo hộ": "Chưa rõ",
            "Trạng thái": "Chưa kiểm tra",
            "is_safe": False,
            "employee_name": "Chưa xác định",
            "stranger_image": ""
        }

    def _load_employee_db(self):
        """Lấy toàn bộ embedding khuôn mặt từ DB lên RAM khi khởi động"""
        db = SessionLocal()
        emp_db = []
        try:
            employees = db.query(Employee).filter(Employee.face_embedding.isnot(None)).all()
            for emp in employees:
                try:
                    emb = np.array(json.loads(emp.face_embedding))
                    emp_db.append((emp.employee_id, emp.name, emb))
                except Exception:
                    pass
        finally:
            db.close()
        print(f"[*] Đã tải {len(emp_db)} hồ sơ khuôn mặt nhân viên vào hệ thống.")
        return emp_db

    def start(self):
        """Mở luồng camera"""
        with self.lock:
            # Ngắt kết nối cũ nếu đang chạy để mở nguồn mới
            if self.is_running:
                self._stop_internal()
                
            print(f"[*] Bắt đầu camera từ nguồn: {self.source}")
            # Nếu source là string số ("0", "1") chuyển thành int để OpenCV nhận diện USB camera
            try:
                src = int(self.source)
            except ValueError:
                src = self.source
                
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                print(f"[!] Không thể mở camera. Kiểm tra lại nguồn: {self.source}")
                return False
                
            self.is_running = True
            self.latest_frame = None
            
            # Khởi động thread chạy ngầm để lấy frame liên tục, tránh đầy buffer của IP Camera
            self.read_thread = threading.Thread(target=self._update_frames, daemon=True)
            self.read_thread.start()
            return True

    def _update_frames(self):
        """Tiểu trình chạy ẩn: Liên tục lấy frame từ camera để chống bị chặn kết nối (Dành riêng IP Camera)"""
        while self.is_running and self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.latest_frame = frame
            else:
                time.sleep(0.01)

    def _stop_internal(self):
        """Dừng camera nhưng không request lock ở vòng ngoài (tránh deadlock)"""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def stop(self):
        """Tắt luồng camera"""
        with self.lock:
            self._stop_internal()
            print("[*] Đã đóng camera.")
            
    def _async_face_scan(self, frame, frame_evaluations):
        """Tiểu trình chạy ẩn: Tránh InsightFace (chạy trên CPU) làm giật lag/khựng hình video feed"""
        faces = app_face.get(frame)
        new_tracks = []
        stranger_img_path = ""
        
        for face in faces:
            f_box = face.bbox
            emb = face.embedding
            
            # Tìm Person Box đang giao cắt
            best_person_idx = -1
            best_overlap = 0
            for idx, p_eval in enumerate(frame_evaluations):
                overlap = get_intersection_ratio(p_eval["box"], f_box)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_person_idx = idx
                    
            if best_overlap > 0.1 and best_person_idx != -1:
                # Nhận diện
                best_match = ("ID_UNKNOWN", "NGƯỜI LẠ")
                best_dist = 999
                for emp_id, emp_name, db_emb in self.employee_db:
                    dist = 1 - np.dot(emb, db_emb) / (norm(emb) * norm(db_emb))
                    if dist < best_dist:
                        best_dist = dist
                        best_match = (emp_id, emp_name)
                        
                if best_match[0] != "ID_UNKNOWN" and best_dist < 0.6:
                    new_tracks.append({"box": frame_evaluations[best_person_idx]["box"], "emp_info": best_match})
                else:
                    new_tracks.append({"box": frame_evaluations[best_person_idx]["box"], "emp_info": ("ID_UNKNOWN", "NGƯỜI LẠ")})
                    
                    # Chụp người lạ
                    if time.time() - self.stranger_cooldown > 3.0:
                        box = f_box.astype(int)
                        x1, y1 = max(0, box[0]), max(0, box[1])
                        x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            fname = f"stranger_{int(time.time())}.jpg"
                            fp = os.path.join(self.snapshot_dir, fname)
                            cv2.imwrite(fp, face_crop)
                            stranger_img_path = f"/static/snapshots/{fname}"
                            self.stranger_cooldown = time.time()
                            
        # Trộn ID mới quét được vào tracked_identities dựa trên IoU
        for nt in new_tracks:
            matched = False
            for tt in self.tracked_identities:
                if get_iou(nt["box"], tt["box"]) > 0.3:
                    tt["emp_info"] = nt["emp_info"]
                    matched = True
                    break
            if not matched:
                self.tracked_identities.append(nt)
                
        if stranger_img_path:
            self.latest_status["stranger_image"] = stranger_img_path

    def _save_log(self, is_safe, details, frame_img=None, emp_info=("ID_UNKNOWN", "Chưa xác định")):
        """Hàm ghi log vi phạm/an toàn vào cơ sở dữ liệu"""
        current_time = time.time()
        
        # Thiết lập Log chống Spam (1 hồ sơ 1 lần báo cáo trong 60 giây, trừ khi đổi trạng thái)
        if not hasattr(self, 'last_log_state'):
            self.last_log_state = {}
            
        person_id = emp_info[0]
        
        # Nếu đã có mặt trong 60 giây qua với CÙNG một trạng thái an toàn -> Bỏ qua, giảm rác log
        if person_id in self.last_log_state:
            last_time, last_safe = self.last_log_state[person_id]
            if current_time - last_time < 60.0 and last_safe == is_safe:
                return # Log throttle
                
        self.last_log_state[person_id] = (current_time, is_safe)
        self.last_log_time = current_time
        
        image_path = None
        if frame_img is not None and not is_safe:
            filename = f"snapshot_{int(current_time)}.jpg"
            filepath = os.path.join(self.snapshot_dir, filename)
            cv2.imwrite(filepath, frame_img)
            image_path = f"/static/snapshots/{filename}"
            
        db = SessionLocal()
        try:
            log = DetectionLog(
                employee_id=emp_info[0], 
                employee_name=emp_info[1],
                is_safe=is_safe,
                details=details,
                image_path=image_path
            )
            db.add(log)
            db.commit()
            print(f"[*] Log saved: {emp_info[1]} - {details}")
        except Exception as e:
            print(f"[!] Lỗi khi lưu log: {e}")
        finally:
            db.close()

    def generate_frames(self):
        """Trình tạo (Generator) MJPEG phục vụ tag <img> qua /video_feed"""
        # Nếu chưa mở thì mở lên
        if not self.is_running or not self.cap or not self.cap.isOpened():
            self.start()

        while self.is_running:
            if self.latest_frame is None:
                # Trả về 1 khung hình đen tạm thời để thông báo luồng HTTP đang trờ dữ liệu thay vì treo cứng
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Connecting to IP Camera... Please wait", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue
                
            # Sao chép frame độc lập để gửi xuống model, đảm bảo cv2.VideoCapture đọc tự do
            frame = self.latest_frame.copy()

            # 1. Rã kết quả của YOLO
            results = self.model.predict(source=frame, stream=True, conf=0.5, verbose=False)
            
            persons = []
            helmets = []
            vests = []
            annotated_frame = frame.copy()
            
            for r in results:
                annotated_frame = r.plot()
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = self.model.names[cls_id].lower()
                        xyxy = box.xyxy[0].cpu().numpy()
                        
                        if class_name in ['person', 'nguoi']:
                            persons.append(xyxy)
                        elif class_name in ['mu', 'helmet', 'hard_hat']:
                            helmets.append(xyxy)
                        elif class_name in ['ao', 'vest', 'safety_vest']:
                            vests.append(xyxy)
            
            # 2. Thuật toán Multi-Object Mapping
            frame_evaluations = []
            
            for p_box in persons:
                has_mu = False
                has_ao = False
                
                # Check Mũ (Nếu có Mũ Overlap > 30% với Người)
                for h_box in helmets:
                    if get_intersection_ratio(p_box, h_box) > 0.3:
                         has_mu = True
                         break
                
                # Check Áo (Nếu có Áo Overlap > 30% với Người)
                for v_box in vests:
                    if get_intersection_ratio(p_box, v_box) > 0.3:
                         has_ao = True
                         break
                         
                frame_evaluations.append({
                    "box": p_box,
                    "has_mu": has_mu,
                    "has_ao": has_ao,
                    "is_safe": has_mu and has_ao,
                    "emp_info": ("ID_UNKNOWN", "NGƯỜI LẠ")
                })
                
            # Duy trì Tên định danh (Tracking Identifier Retention giữa các Frames)
            for p_eval in frame_evaluations:
                best_track = None
                best_iou = 0
                for track in self.tracked_identities:
                    iou = get_iou(p_eval["box"], track["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_track = track
                        
                if best_iou > 0.3 and best_track is not None:
                    p_eval["emp_info"] = best_track["emp_info"]
            
            # 3. Quét khuôn mặt Async (Chạy luồng riêng lẻ để không giật lag màn hình)
            self.frame_counter += 1
            if self.frame_counter % 10 == 0:
                threading.Thread(target=self._async_face_scan, args=(frame.copy(), frame_evaluations.copy()), daemon=True).start()
                                    
            # Refresh lại danh sách Tracking với các tọa độ mới nhất của Frame hiện tại bằng IOU
            active_tracks = []
            for p_eval in frame_evaluations:
                # Mặc định lấy theo config cũ (từ logic IOo ở #2)
                active_tracks.append({"box": p_eval["box"], "emp_info": p_eval["emp_info"]})
            self.tracked_identities = active_tracks
            
            # 4. In thông tin lên màn hình và Log xuống DB cho TỪNG người riêng biệt
            for p_eval in frame_evaluations:
                box = p_eval["box"].astype(int)
                emp_name = p_eval["emp_info"][1]
                missing_items = []
                if not p_eval["has_mu"]: missing_items.append("Mũ")
                if not p_eval["has_ao"]: missing_items.append("Áo")
                
                is_safe = p_eval["is_safe"]
                
                if not is_safe:
                    text_status = "Vi Pham: " + ", ".join(missing_items)
                    color = (0, 0, 255)
                else:
                    text_status = "An Toan"
                    color = (0, 255, 0)
                    
                # Nhúng thông tin trực tiếp trôi theo đầu người
                cv2.putText(annotated_frame, f"ID: {emp_name}", (int(box[0]), int(max(0, box[1] - 35))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, text_status, (int(box[0]), int(max(0, box[1] - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                # Gọi hệ thống xuất Log đa luồng
                self._save_log(is_safe, ", ".join(missing_items) if not is_safe else "Đầy đủ PPE", annotated_frame, emp_info=p_eval["emp_info"])

            # 5. Cập nhật Status Dashboard dựa trên Bounding Box to nhất
            if len(frame_evaluations) > 0:
                largest_p = sorted(frame_evaluations, key=lambda x: (x["box"][2]-x["box"][0])*(x["box"][3]-x["box"][1]), reverse=True)[0]
                
                self.latest_status["Áo bảo hộ"] = "Detected" if largest_p["has_ao"] else "None"
                self.latest_status["Mũ bảo hộ"] = "Detected" if largest_p["has_mu"] else "None"
                self.latest_status["Giày bảo hộ"] = "Chưa rõ" 
                
                if largest_p["is_safe"]:
                    self.latest_status["Trạng thái"] = "Hoàn Toàn An Toàn"
                    self.latest_status["is_safe"] = True
                else:
                    self.latest_status["Trạng thái"] = "Vi Phạm PPE"
                    self.latest_status["is_safe"] = False
                    
                self.latest_status["employee_name"] = largest_p["emp_info"][1]
            else:
                self.latest_status["Áo bảo hộ"] = "None"
                self.latest_status["Mũ bảo hộ"] = "None"
                self.latest_status["Trạng thái"] = "Chưa kiểm tra"
                self.latest_status["is_safe"] = False
                self.latest_status["employee_name"] = "Không có người"
                
            # LUÔN MÃ HÓA VÀ GỬI ẢNH XUỐNG DÙ CÓ NGƯỜI HAY KHÔNG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                # Cấu trúc bytes stream gửi qua HTTP x-mixed-replace
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except GeneratorExit:
                    self.stop()
                    raise

from config import CAMERA_SOURCE

camera_system = CameraStream(source=CAMERA_SOURCE)
