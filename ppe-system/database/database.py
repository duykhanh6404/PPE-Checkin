from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import os

# Tạo thư mục data nếu chưa có
os.makedirs("data", exist_ok=True)
SQLALCHEMY_DATABASE_URL = "sqlite:///./data/ppe_system.db"

# Khởi tạo engine cho SQLite
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Bảng danh sách nhân viên để phục vụ Face Scan
class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, unique=True, index=True) # Mã nhân viên (ví dụ NV001)
    name = Column(String)                                 # Tên nhân viên
    face_embedding = Column(String, nullable=True)        # Đặc trưng khuôn mặt (JSON string chứa mảng float)
    created_at = Column(DateTime, default=datetime.datetime.now)

# Bảng log lịch sử quét PPE mỗi khi có người đi vào
class DetectionLog(Base):
    __tablename__ = "detection_logs"
    id = Column(Integer, primary_key=True, index=True)
    employee_id = Column(String, index=True)              # Mã nhân viên (nếu nhận diện mặt) hoặc UNKNOWN
    employee_name = Column(String)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    is_safe = Column(Boolean, default=False)              # Đủ PPE hay không
    details = Column(String)                              # Thiếu gì? Ví dụ: Không mũ, không áo
    image_path = Column(String, nullable=True)            # Ảnh lưu lại làm chứng cứ

# Tạo toàn bộ bảng khi file này được import lần đầu
Base.metadata.create_all(bind=engine)

# Migration: Thêm cột face_embedding nếu chưa có (tránh lỗi trên CSDL cũ)
try:
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE employees ADD COLUMN face_embedding TEXT"))
except Exception as e:
    pass # Cột đã tồn tại hoặc có lỗi khác, bỏ qua

# Dependency cung cấp session thao tác database cho FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
