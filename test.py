from ultralytics import YOLO
import cv2

# Đường dẫn đến file weights (mô hình) đã được huấn luyện tốt nhất
# best.pt nằm cùng thư mục nên chỉ cần truyền tên file
model_path = "best.pt"

print(f"Đang nạp mô hình từ {model_path}...")
model = YOLO(model_path)

# Nạp file config từ hệ thống chính ppe-system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'ppe-system'))
from config import CAMERA_SOURCE

# Cấu hình nguồn camera: 
# Đồng bộ theo cấu hình trung tâm (config.py)
camera_source = CAMERA_SOURCE
print(f"Bắt đầu kết nối với camera laptop...")

# Thực hiện inference liên tục từ luồng camera
# Cấu hình các tham số:
#   source: Nguồn hình ảnh (ở đây là webcam "0")
#   show=True: Tự động dùng OpenCV để hiển thị một cửa sổ chứa kết quả dự đoán (có vẽ bounding boxes)
#   stream=True: Hoạt động như một Generator để xử lý dữ liệu trực tiếp, không làm tràn RAM khi chạy video dài
#   conf=0.5: Ngưỡng tin cậy (confidence threshold) - chỉ hiện những khung hình / dự đoán có độ chính xác > 50%
results = model.predict(source=camera_source, show=True, stream=True, conf=0.5)

# Chúng ta cần lặp qua trình tạo (generator) để mô hình thực sự chạy trên từng khung hình của webcam
for result in results:
    # Các kết quả sẽ được vẽ và hiển thị trên màn hình tự động do tham số show=True.
    
    # Nếu bạn cần lấy thông tin chi tiết của các đối tượng nhận diện được ở mỗi khung hình để lập trình xử lý tiếp:
    # boxes = result.boxes  # Bao gồm tọa độ (xyxy, xywh), độ tin cậy (conf), class ID (cls)
    # class_names = result.names # Dictionary chứa tên các lớp (ví dụ: {0: 'mask', 1: 'no_mask'})
    pass
