from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
import shutil

from routes import dashboard, history, facescan, report

# Khởi tạo instance của FastAPI Backend
app = FastAPI(title="PPE Detection - YOLO11 Server", description="Hệ thống Camera quét đồ bảo hộ an toàn lao động")

# Cấu hình CORS - Cho phép gọi API giữa giao diện Frontend tĩnh và Backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nhúng tất cả các routes được khai báo logic ở nhóm thư mục ./routes/
app.include_router(dashboard.router, tags=["Dashboard Camera API"])
app.include_router(history.router, tags=["History Logs API"])
app.include_router(facescan.router, tags=["Face Scanner API"])
app.include_router(report.router, tags=["Report Excel API"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

print("[*] Đang khởi động tiến trình chuẩn bị frontend...")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

parent_static = os.path.join(os.path.dirname(BASE_DIR), "static")
if os.path.isdir(parent_static) and len(os.listdir(STATIC_DIR)) <= 2: 
    print("[*] Đang tự động sao chép các tệp HTML/CSS/JS từ dự án ../static sang ./static/...")
    # Lặp chép thư mục
    import shutil
    for item in os.listdir(parent_static):
        s = os.path.join(parent_static, item)
        d = os.path.join(STATIC_DIR, item)
        if os.path.isfile(s) and not os.path.exists(d):
            shutil.copy2(s, d)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    print("=========================================================================")
    print("[*] Server FastAPI + YOLO11 Backend đang khởi động...")
    print("[*] Tru cập ứng dụng Fullstack ngay tại: http://localhost:8000")
    print("=========================================================================")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
