import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

_model = None

def get_yolo_model():
    """Tải và trả về mô hình YOLO11 duy nhất phục vụ ứng dụng"""
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            print(f"[*] Đang tải mô hình YOLO11 từ {MODEL_PATH}...")
            _model = YOLO(MODEL_PATH)
        else:
            # Fallback nếu chưa có best.pt (dùng pre-trained yolo11n.pt mặc định)
            print(f"[!] WARNING: Không tìm thấy {MODEL_PATH}. Tải mô hình mặc định yolo11n.pt để thử nghiệm.")
            _model = YOLO("yolo11n.pt")
    return _model
