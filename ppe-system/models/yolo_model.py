import os
from ultralytics import YOLO

# Đường dẫn mặc định của file mô hình tốt nhất (YOLO11 weights)
# Vì file best.pt đang nằm trong thư mục models/ nên ta trỏ đường dẫn vào đó.
MODEL_PATH = "models/best.pt"

# Biến global lưu trữ thể hiện (instance) của YOLO để tránh tải lại (reload) Model liên tục
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
