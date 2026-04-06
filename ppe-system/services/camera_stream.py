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
        
        # Đặc trưng khuôn mặt NV
        self.employee_db = self._load_employee_db()
        self.frame_counter = 0
        self.latest_person_matched = ("ID_UNKNOWN", "Chưa xác định")
        self.stranger_cooldown = 0
        
        # Lock để đảm bảo an toàn nếu nhiều người cùng truy cập xem ảnh
        self.lock = threading.Lock()
        
        # Thư mục chứa ảnh bị lỗi PPE
        self.snapshot_dir = "static/snapshots"
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
                time.sleep(0.05)
                continue
                
            # Sao chép frame độc lập để gửi xuống model, đảm bảo cv2.VideoCapture đọc tự do
            frame = self.latest_frame.copy()

            # 1. Nhận diện khuôn mặt cứ mỗi 5 Frames (tránh giật FPS)
            self.frame_counter += 1
            if self.frame_counter % 5 == 0:
                faces = app_face.get(frame)
                if len(faces) > 0:
                    # Lấy khuôn mặt có diện tích lớn nhất (gần camera nhất)
                    face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
                    emb = face.embedding
                    
                    best_match = None
                    best_dist = 999
                    for emp_id, emp_name, db_emb in self.employee_db:
                        # Cosine distance
                        dist = 1 - np.dot(emb, db_emb) / (norm(emb) * norm(db_emb))
                        if dist < best_dist:
                            best_dist = dist
                            best_match = (emp_id, emp_name)
                    
                    if best_match and best_dist < 0.6:
                        # Người quen
                        self.latest_person_matched = best_match
                        self.latest_status["stranger_image"] = ""
                    else:
                        # Người lạ
                        self.latest_person_matched = ("ID_UNKNOWN", "NGƯỜI LẠ")
                        if time.time() - self.stranger_cooldown > 3.0:
                            # Cắt khung hình người lạ lưu vào thư mục
                            box = face.bbox.astype(int)
                            x1, y1 = max(0, box[0]), max(0, box[1])
                            x2, y2 = min(frame.shape[1], box[2]), min(frame.shape[0], box[3])
                            face_crop = frame[y1:y2, x1:x2]
                            
                            if face_crop.size > 0:
                                fname = f"stranger_{int(time.time())}.jpg"
                                fp = os.path.join(self.snapshot_dir, fname)
                                cv2.imwrite(fp, face_crop)
                                self.latest_status["stranger_image"] = f"/static/snapshots/{fname}"
                                self.stranger_cooldown = time.time()

            # Gán lại thông tin định danh mới nhất vào Status
            self.latest_status["employee_name"] = self.latest_person_matched[1]

            # 2. Thực hiện detect Đồ Bảo Hộ với best.pt YOLO (Chạy stream=True, 0.5)
            results = self.model.predict(source=frame, stream=True, conf=0.5, verbose=False)
            
            is_safe = True
            missing_items = []
            
            for r in results:
                # YOLO có hàm r.plot() tự động vẽ Bounding box vào ảnh
                annotated_frame = r.plot()
                
                detected_classes = []
                if len(r.boxes) > 0:
                    for cls_id in r.boxes.cls:
                        class_name = self.model.names[int(cls_id)].lower()
                        detected_classes.append(class_name)
                
                # Check theo mảng lớp PPE (đã xoá kính)
                has_ao = any(x in detected_classes for x in ['ao', 'vest', 'ao_bao_ho', 'safety_vest'])
                has_mu = any(x in detected_classes for x in ['mu', 'helmet', 'mu_bao_ho', 'hard_hat'])
                has_giay = any(x in detected_classes for x in ['giay', 'shoe', 'giay_bao_ho', 'safety_shoes'])
                
                has_no_ao = any(x in detected_classes for x in ['no_vest', 'noao', 'khong_ao', 'no-vest'])
                has_no_mu = any(x in detected_classes for x in ['no_helmet', 'nomu', 'khong_mu', 'no-helmet'])
                has_no_giay = any(x in detected_classes for x in ['no_shoes', 'nogiay', 'khong_giay'])
                
                self.latest_status["Áo bảo hộ"] = "Detected" if (has_ao and not has_no_ao) else "None"
                self.latest_status["Mũ bảo hộ"] = "Detected" if (has_mu and not has_no_mu) else "None"
                self.latest_status["Giày bảo hộ"] = "Detected" if (has_giay and not has_no_giay) else "None"
                
                # Status tổng quát (bỏ kính, bỏ giày ra khỏi danh sách cấm)
                if len(detected_classes) > 0:
                    # BẮT BUỘC phải phát hiện ra Áo và Mũ thì mới An Toàn
                    if has_ao and has_mu:
                        is_safe = True
                        missing_items = []
                        self.latest_status["Trạng thái"] = "Hoàn Toàn An Toàn"
                        self.latest_status["is_safe"] = True
                    else:
                        is_safe = False
                        if not has_ao: missing_items.append("Áo")
                        if not has_mu: missing_items.append("Mũ")
                        # Giày lỗi không bắt buộc, không dính lỗi
                        self.latest_status["Trạng thái"] = "Vi Phạm PPE"
                        self.latest_status["is_safe"] = False
                else:
                    is_safe = True
                    missing_items = []
                    self.latest_status["Trạng thái"] = "Chưa kiểm tra"
                    self.latest_status["is_safe"] = False
                
                # Ghi đè tên nhân viên / Người lạ lên màn hình Video
                cv2.putText(annotated_frame, f"ID: {self.latest_person_matched[1]}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
                
                if not is_safe:
                    text_status = "Khong an toan: " + ", ".join(set(missing_items))
                    color = (0, 0, 255) # Đỏ
                else:
                    text_status = "An toan"
                    color = (0, 255, 0) # Xanh lá
                
                cv2.putText(annotated_frame, text_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
                
                # Lưu lịch sử log đính kèm danh tính NV
                if not is_safe:
                    self._save_log(False, ", ".join(set(missing_items)), annotated_frame, emp_info=self.latest_person_matched)
                else:
                    if len(detected_classes) > 0:
                        self._save_log(True, "Đầy đủ PPE", None, emp_info=self.latest_person_matched)
                
                # Mã hóa ảnh thành chuẩn JPEG gửi xuống trình duyệt (MJPEG bytes)
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    # Cấu trúc bytes stream gửi qua HTTP x-mixed-replace
                    try:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    except GeneratorExit:
                        self.stop() # Tự động ngắt kết nối DroidCam khi client (user) đóng/reload tab trình duyệt
                        raise

from config import CAMERA_SOURCE

# Global instance phục vụ singleton model và stream cho Router FastAPI
# Thiết lập nguồn từ config.py trung tâm
camera_system = CameraStream(source=CAMERA_SOURCE)
