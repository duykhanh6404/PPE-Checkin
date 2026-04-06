from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from services.camera_stream import camera_system

# Gắn prefix để gọi API qua JS, VD: fetch('/api/dashboard/camera/start')
router = APIRouter(prefix="/api/dashboard")

class CameraConfig(BaseModel):
    source: str = "" 

@router.get("/video_feed")
def video_feed():
    """Endpoint quan trọng nhất trả hình ảnh qua trình duyệt bằng x-mixed-replace MJPEG stream"""
    return StreamingResponse(
        camera_system.generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.post("/camera/start")
def start_camera(config: CameraConfig):
    """Nút BẮT ĐẦU từ giao diện Front-end, thay đổi URL luồng video mới và chạy stream"""
    if config and config.source:
        camera_system.source = config.source
    
    # Khởi động camera logic
    success = camera_system.start()
    if success:
        return {"status": "success", "message": "Đã bắt đầu IP camera thành công."}
    else:
        return {"status": "error", "message": "Lỗi: Không thể kết nối tới nguồn camera này!"}

@router.post("/camera/stop")
def stop_camera():
    """Nút DỪNG kết nối luồng stream"""
    camera_system.stop()
    return {"status": "success", "message": "Camera đã ngừng hoạt động."}

@router.get("/camera/status")
def get_status():
    """Lấy trạng thái hệ thống: đang check, nguồn,..."""
    return {
        "status": "success",
        "is_running": camera_system.is_running,
        "source": camera_system.source
    }

@router.get("/current_status")
def get_current_status():
    """Lấy kết quả quét PPE mới nhất từ model vừa update"""
    return camera_system.latest_status