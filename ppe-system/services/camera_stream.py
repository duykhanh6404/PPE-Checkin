import cv2
import time
import os
import threading
from models.yolo_model import get_yolo_model
from models.face_model import app_face
from database.database import SessionLocal, DetectionLog, Employee
import datetime
import json
import re
import numpy as np
from numpy.linalg import norm
import faiss

def remove_vietnamese_accents(s):
    s = str(s)
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s

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
    """Class xử lý stream camera, detect YOLO realtime và log lịch sử theo kiến trúc Producer-Consumer"""
    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.is_running = False
        self.model = get_yolo_model()
        
        # Threads
        self.read_thread = None
        self.ai_thread = None
        
        # Memory Shared Lock-free (LIFO Frames)
        self.latest_raw_frame = None
        self.latest_annotated_frame = None
        
        self.employee_db = []
        self.faiss_index = None
        self._load_employee_db()
        
        self.frame_counter = 0
        self.stranger_cooldown = 0
        self.stranger_id_counter = 0
        self.tracked_identities = [] # Tracking ID Boxes
        
        self.lock = threading.Lock()
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.snapshot_dir = os.path.join(BASE_DIR, "static", "snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        self.last_log_time = 0
        
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
        db = SessionLocal()
        emp_list = []
        vectors = []
        try:
            employees = db.query(Employee).filter(Employee.face_embedding.isnot(None)).all()
            for emp in employees:
                try:
                    emb = np.array(json.loads(emp.face_embedding), dtype=np.float32)
                    emp_list.append((emp.employee_id, emp.name))
                    vectors.append(emb)
                except Exception:
                    pass
        finally:
            db.close()
            
        print(f"[*] Đã tải {len(emp_list)} hồ sơ khuôn mặt nhân viên vào hệ thống.")
        
        self.employee_db = emp_list
        # Khởi tạo FAISS Vector Index thay vì lưu Array thô
        if len(vectors) > 0:
            dim = vectors[0].shape[0]  # Thường là 512
            self.faiss_index = faiss.IndexFlatIP(dim) # COSINE Similarity cần L2 Normalized + Inner Product Index
            
            # Khởi tạo ma trận (N x 512) và chuẩn hóa phân bổ (L2)
            emb_matrix = np.vstack(vectors).astype(np.float32)
            faiss.normalize_L2(emb_matrix)
            
            self.faiss_index.add(emb_matrix)
            print(f"[*] Đã biên dịch {len(vectors)} vector khuôn mặt vào FAISS Core Database để tăng tốc truy vấn!")
        else:
            self.faiss_index = None

    def update_embeddings(self):
        """Hot-reload: Tải lại cơ sở dữ liệu khuôn mặt ngay lập tức vào RAM mà không cần restart"""
        with self.lock:
            self._load_employee_db()
            print("[*] Hot-reload thành công: AI đã cập nhật bộ nhớ nhận diện FAISS VectorDB.")

    def start(self):
        with self.lock:
            if self.is_running:
                self._stop_internal()
                
            print(f"[*] Bắt đầu camera từ nguồn: {self.source}")
            try:
                src = int(self.source)
            except ValueError:
                src = self.source
                
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                print(f"[!] Không thể mở camera. Kiểm tra lại nguồn: {self.source}")
                return False
                
            self.is_running = True
            
            self.latest_raw_frame = None
            self.latest_annotated_frame = None
            
            # [PRODUCER] Luồng 1: Chỉ bắt khung hình từ Camera
            self.read_thread = threading.Thread(target=self._camera_read_worker, daemon=True)
            self.read_thread.start()
            
            # [CONSUMER 1] Luồng 2: AI YOLO xử lý
            self.ai_thread = threading.Thread(target=self._ai_inference_worker, daemon=True)
            self.ai_thread.start()
            
            return True

    def _camera_read_worker(self):
        """Tiểu trình 1: Producer - Liên tục lấy ảnh thô từ Camera nhanh nhất có thể (30-60 FPS)"""
        while self.is_running and self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.latest_raw_frame = frame
            else:
                time.sleep(0.01)

    def _stop_internal(self):
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def stop(self):
        with self.lock:
            self._stop_internal()
            print("[*] Đã đóng camera.")
            
    def _async_face_scan(self, frame, frame_evaluations):
        """Tiểu trình nhánh AI: Quét khuôn mặt dưới nền không làm block YOLO hay HTTP"""
        faces = app_face.get(frame)
        new_tracks = []
        stranger_img_path = ""
        
        for face in faces:
            f_box = face.bbox
            emb = face.embedding
            
            best_person_idx = -1
            best_overlap = 0
            for idx, p_eval in enumerate(frame_evaluations):
                overlap = get_intersection_ratio(p_eval["box"], f_box)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_person_idx = idx
                    
            if best_overlap > 0.1 and best_person_idx != -1:
                best_match = ("ID_UNKNOWN", "NGƯỜI LẠ")
                best_dist = 999.0
                
                if self.faiss_index is not None and self.faiss_index.ntotal > 0:
                    # Truy vấn 1 tỉ lệ vector bằng FAISS Matrix (Cực nhanh O(1))
                    emb_query = emb.astype(np.float32).reshape(1, -1)
                    faiss.normalize_L2(emb_query)
                    
                    distances, indices = self.faiss_index.search(emb_query, 1) # Lấy k=1 (người giống nhất)
                    best_idx = indices[0][0]
                    # IndexFlatIP trả về Cosine Similarity. Do hệ thống so khoảng cách lỗi, ta đổi: Distance = 1.0 - Similarity
                    best_dist = 1.0 - distances[0][0] 
                    
                    if best_dist < 0.6:
                        best_match = self.employee_db[best_idx]
                        
                if best_match[0] != "ID_UNKNOWN" and best_dist < 0.6:
                    new_tracks.append({"box": frame_evaluations[best_person_idx]["box"], "emp_info": best_match})
                else:
                    # KHÔNG chèn chuỗi ("ID_UNKNOWN", "NGƯỜI LẠ") cứng vào new_tracks nữa.
                    # Phải giữ nguyên ID_UNKNOWN_XXX do IoU Tracker đã cấp phát!
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
        current_time = time.time()
        
        if not hasattr(self, 'last_log_state'):
            self.last_log_state = {}
            
        person_id = emp_info[0]
        
        if person_id in self.last_log_state:
            last_time, last_safe = self.last_log_state[person_id]
            if current_time - last_time < 60.0 and last_safe == is_safe:
                return
                
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

    def _ai_inference_worker(self):
        """Tiểu trình 2: AI Consumer - Hút ảnh từ raw, nhai model YOLO, tống xuống latest_annotated_frame"""
        last_processed_id = None
        
        while self.is_running:
            if self.latest_raw_frame is None:
                time.sleep(0.02)
                continue
                
            frame = self.latest_raw_frame
            # Chỉ xử lý nếu có frame mới tới
            if id(frame) == last_processed_id:
                time.sleep(0.01)
                continue
                
            last_processed_id = id(frame)
            frame_copy = frame.copy()
            
            results = self.model.predict(source=frame_copy, stream=True, conf=0.5, verbose=False)
            
            persons = []
            helmets = []
            vests = []
            annotated_frame = frame_copy.copy()
            
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
            
            frame_evaluations = []
            for p_box in persons:
                has_mu = False
                has_ao = False
                
                for h_box in helmets:
                    if get_intersection_ratio(p_box, h_box) > 0.3:
                         has_mu = True
                         break
                
                for v_box in vests:
                    if get_intersection_ratio(p_box, v_box) > 0.3:
                         has_ao = True
                         break
                         
                frame_evaluations.append({
                    "box": p_box,
                    "has_mu": has_mu,
                    "has_ao": has_ao,
                    "is_safe": has_mu and has_ao,
                    "emp_info": None
                })
                
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
                else:
                    self.stranger_id_counter += 1
                    p_eval["emp_info"] = (f"ID_UNKNOWN_{self.stranger_id_counter}", "NGƯỜI LẠ")
            
            self.frame_counter += 1
            if self.frame_counter % 10 == 0:
                threading.Thread(target=self._async_face_scan, args=(frame_copy.copy(), frame_evaluations.copy()), daemon=True).start()
                                    
            active_tracks = []
            for p_eval in frame_evaluations:
                active_tracks.append({"box": p_eval["box"], "emp_info": p_eval["emp_info"]})
            self.tracked_identities = active_tracks
            
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
                    log_details = "Thiếu: " + ", ".join(missing_items)
                else:
                    text_status = "An Toan"
                    color = (0, 255, 0)
                    log_details = "Đầy đủ PPE"
                    
                # Chuyển đổi chuỗi tiếng Việt có dấu thành không dấu riêng cho khung hình OpenCV để tránh lỗi font ASCII
                clean_name = remove_vietnamese_accents(emp_name)
                clean_status = remove_vietnamese_accents(text_status)
                    
                cv2.putText(annotated_frame, f"ID: {clean_name}", (int(box[0]), int(max(0, box[1] - 35))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
                cv2.putText(annotated_frame, clean_status, (int(box[0]), int(max(0, box[1] - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
                
                self._save_log(is_safe, log_details, annotated_frame, emp_info=p_eval["emp_info"])

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
                
            # Đẩy ảnh hoàn chỉnh ra hệ thống Web
            self.latest_annotated_frame = annotated_frame

    def generate_frames(self):
        """Tiểu trình 3: Server Consumer - Chuyên phân phối ảnh MJPEG, siêu tĩnh và nhẹ (30-60FPS)"""
        if not self.is_running or not self.cap or not self.cap.isOpened():
            self.start()

        while self.is_running:
            # Ưu tiên lấy ảnh đã được AI vẽ. Nếu AI chưa kịp vẽ thì lấy frame gốc chống nghẽn
            frame_to_yield = self.latest_annotated_frame
            if frame_to_yield is None:
                frame_to_yield = self.latest_raw_frame
                
            if frame_to_yield is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Connecting to IP Camera... Please wait", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', blank)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue
                
            ret, buffer = cv2.imencode('.jpg', frame_to_yield)
            if ret:
                frame_bytes = buffer.tobytes()
                try:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except GeneratorExit:
                    self.stop()
                    raise
            
            # Cố định Web Stream giới hạn ở tỷ lệ làm tươi tối đa 30 FPS.
            time.sleep(1/30.0)

from config import CAMERA_SOURCE

camera_system = CameraStream(source=CAMERA_SOURCE)
