from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
import cv2
import os
import time
from database.database import SessionLocal, Employee
import numpy as np
import json
from models.face_model import app_face

router = APIRouter(prefix="/api/facescan")

# Import hệ thống camera dùng chung để không chiếm dụng luồng IP Droidcam độc quyền 
from services.camera_stream import camera_system

# Dữ liệu truyền từ JS front-end sang
class ScanRequest(BaseModel):
    employee_id: str
    employee_name: str

# Biến toàn cục lưu trạng thái quét để chia sẻ giữa Background Task và Video Stream
scan_state = {
    "is_scanning": False,
    "message": "",
    "color": (0, 255, 0),
    "progress": 0
}

def capture_face_task(employee_id: str, employee_name: str):
    """Tiến trình ngầm Smart Scanner: Hướng dẫn người dùng quét các góc quay của khuôn mặt"""
    global scan_state
    
    faces_dir = f"static/faces/{employee_id}"
    os.makedirs(faces_dir, exist_ok=True)
    
    print(f"[*] Bắt đầu Scanner khuôn mặt: {employee_id} - {employee_name}...")
    
    # Chắc chắn rằng camera trung tâm (từ dashboard/config) đang mở
    if not camera_system.is_running:
        camera_system.start()
    
    scan_state["is_scanning"] = True
    scan_state["progress"] = 0
    scan_state["message"] = "Bat dau! Hay dua mat vao vong tron"
    scan_state["color"] = (0, 255, 255) # Vàng
    
    embeddings = []
    
    phases = [
        {"name": "NHIN THANG", "condition": lambda yaw: -15 <= yaw <= 15, "frames_needed": 3},
        {"name": "QUAY SANG TRAI", "condition": lambda yaw: yaw < -2, "frames_needed": 3},
        {"name": "QUAY SANG PHAI", "condition": lambda yaw: yaw > 2, "frames_needed": 3}
    ]
    
    current_phase = 0
    frames_captured_in_phase = 0
    total_frames_needed = sum(p["frames_needed"] for p in phases)
    total_captured = 0
    
    # Vòng lặp lấy khung hình
    while current_phase < len(phases) and camera_system.is_running:
        frame = camera_system.latest_frame
        if frame is not None:
            frame_copy = frame.copy()
            faces = app_face.get(frame_copy)
            
            h, w = frame_copy.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            if len(faces) == 0:
                scan_state["message"] = "Khong tim thay khuon mat nao!"
                scan_state["color"] = (0, 0, 255)
            elif len(faces) > 1:
                scan_state["message"] = "Phat hien nhieu nguoi! Chi de 1 nguoi."
                scan_state["color"] = (0, 0, 255)
            else:
                face = faces[0]
                box = face.bbox
                face_cx = (box[0] + box[2]) / 2
                face_cy = (box[1] + box[3]) / 2
                
                # Kiểm tra khuôn mặt có nằm trong vùng quét trung tâm không
                dist = np.sqrt((face_cx - center_x)**2 + (face_cy - center_y)**2)
                
                if dist > 120: 
                    scan_state["message"] = "Hay dua khuon mat vao giua vong tron!"
                    scan_state["color"] = (0, 165, 255) # Cam
                else:
                    yaw, pitch, roll = face.pose
                    phase = phases[current_phase]
                    
                    if phase["condition"](yaw):
                        # Khớp góc độ! Tiến hành chụp
                        scan_state["message"] = f"Giu nguyen! Dang chup {phase['name']}..."
                        scan_state["color"] = (0, 255, 0)
                        
                        embeddings.append(face.embedding)
                        file_path = os.path.join(faces_dir, f"frame_{total_captured}.jpg")
                        cv2.imwrite(file_path, frame_copy)
                        
                        frames_captured_in_phase += 1
                        total_captured += 1
                        scan_state["progress"] = int((total_captured / total_frames_needed) * 100)
                        
                        if frames_captured_in_phase >= phase["frames_needed"]:
                            current_phase += 1
                            frames_captured_in_phase = 0
                            
                        # Chờ 0.5s giữa mỗi frame chụp để lấy độ sai lệch tự nhiên
                        time.sleep(0.5)
                    else:
                        scan_state["message"] = f"Vui long {phase['name']} (Goc: {int(yaw)})"
                        scan_state["color"] = (0, 255, 255)

            time.sleep(0.1)
        else:
            time.sleep(0.1)
            
    # Kết thúc quá trình quét
    scan_state["is_scanning"] = False
    scan_state["progress"] = 100
    print(f"[*] Đã hoàn thành bộ ảnh nội suy khuôn mặt vào dữ liệu nhân viên {employee_id}")
    
    # Lấy vector đại diện trung bình để tăng độ chính xác
    final_embedding_json = None
    if len(embeddings) > 0:
        avg_embedding = np.mean(embeddings, axis=0) # Trung bình của các vector
        final_embedding_json = json.dumps(avg_embedding.tolist())
    else:
        print("[!] Cảnh báo: Không thể trích xuất đặc trưng khuôn mặt hợp lệ (có thể ko tìm thấy mặt hoặc có quá nhiều mặt ở frame).")

    # Save thông tin vào DB sqlite
    db = SessionLocal()
    try:
        emp = db.query(Employee).filter(Employee.employee_id == employee_id).first()
        # Insert hay Update
        if not emp:
            emp = Employee(employee_id=employee_id, name=employee_name, face_embedding=final_embedding_json)
            db.add(emp)
        else:
            emp.name = employee_name
            if final_embedding_json:
                emp.face_embedding = final_embedding_json
        db.commit()
    finally:
        db.close()
        
    # Làm mới danh sách bộ nhớ tạm (RAM) của AI ở Dashboard để nhận diện được người vừa quét
    camera_system.employee_db = camera_system._load_employee_db()
    print("[*] Đã tự động tải lại bộ nhớ AI nhận diện tĩnh thành công.")
from fastapi.responses import StreamingResponse

def generate_frames():
    """Hàm đọc luồng stream liên tục từ camera trung tâm để livestream lên HTML qua IP/Webcam"""
    global scan_state
    
    if not camera_system.is_running:
        camera_system.start()
        
    try:
        while camera_system.is_running:
            if camera_system.latest_frame is None:
                time.sleep(0.05)
                continue
                
            # Flip ảnh ngang (Mirror) ngay tại Backend để làm Gương soi, 
            # tránh việc Frontend dùng CSS flip làm lật luôn cả chữ hướng dẫn.
            frame = cv2.flip(camera_system.latest_frame.copy(), 1)
            
            # Nếu đang trong chế độ QUÉT, vẽ hướng dẫn (HUD) lên màn hình
            if scan_state["is_scanning"]:
                h, w = frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                
                # Vẽ vòng tròn căn giữa khuôn mặt
                cv2.circle(frame, (center_x, center_y), 150, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, scan_state["message"], (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, scan_state["color"], 2, cv2.LINE_AA)
                cv2.putText(frame, f"Tien trinh: {scan_state['progress']}%", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "San sang quet", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Mã hóa tĩnh lên HTML
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                # Bắn dữ liệu liên tục theo định dạng multipart
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except GeneratorExit:
        print("[*] Dashboard Facescan vừa ngắt kết nối trình duyệt.")
        pass

@router.get("/stream")
def video_feed():
    """API trả hình ảnh trực tiếp lên giao diện Facescan"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@router.get("/scan_status")
def get_scan_status():
    """API cho Frontend lấy tiến độ render thanh Progress Bar"""
    return {
        "is_scanning": scan_state["is_scanning"],
        "progress": scan_state["progress"]
    }
@router.post("/start_scan")
def start_scan(request: ScanRequest, background_tasks: BackgroundTasks):
    """API gọi chức năng Smart Scan quét mặt nhân sự mới"""
    # Giao lại việc nặng cho background
    background_tasks.add_task(capture_face_task, request.employee_id, request.employee_name)
    
    return {
        "status": "success",
        "message": f"Hệ thống đang tiến hành bật luồng camera để quyét 10 khuôn mặt cho nhân viên {request.employee_name}. Vui lòng nhìn camera!",
        "employee_id": request.employee_id
    }
