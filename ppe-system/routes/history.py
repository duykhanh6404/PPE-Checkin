from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.database import get_db, DetectionLog

router = APIRouter(prefix="/api/history")

@router.get("")
@router.get("/")
def get_history(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """API Fetch dữ liệu vào bảng table theo thứ tự mới nhất (Lịch sử chấm công / quét PPE)"""
    # Order By ID/Thời gian theo chiều Descending
    logs = db.query(DetectionLog).order_by(DetectionLog.timestamp.desc()).offset(skip).limit(limit).all()
    
    return [
        {
            "id": l.id,
            "employee_id": l.employee_id or "UNKNOWN",
            "employee_name": l.employee_name or "Chưa xác định",
            # Xử lý datetime an toàn để tránh sập API nếu timestamp bị lưu dạng chuỗi hoặc rỗng
            "timestamp": l.timestamp.strftime("%d/%m/%Y %H:%M:%S") if hasattr(l.timestamp, "strftime") else str(l.timestamp),
            "is_safe": bool(l.is_safe),
            "details": l.details or "-",
            "image_path": l.image_path or ""
        }
        for l in logs
    ]
