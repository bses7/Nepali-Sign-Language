from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from app.api.deps import get_current_user
from app.schemas.user import User
from app.services.recognition_service import practice_manager
from app.db.session import get_db
import json

from app.services import notification_service

router = APIRouter()

@router.websocket("/ws/{target_char}")
async def practice_websocket(websocket: WebSocket, target_char: str):
    await websocket.accept()
    session_id = str(id(websocket))
    
    try:
        while True:
            data = await websocket.receive_text()
            
            result = practice_manager.process_frame(session_id, data, target_char)
            
            await websocket.send_json(result)
            
            if result["status"] == "completed":
                report = practice_manager.generate_report(session_id, target_char)
                await websocket.send_json({"type": "final_report", "report": report})
                break

    except WebSocketDisconnect:
        print(f"Practice session {session_id} ended.")
    finally:
        if session_id in practice_manager.sessions:
            del practice_manager.sessions[session_id]


@router.post("/save-results")
def save_practice_results(
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    try:
        if not current_user.stats:
             return {"success": False, "message": "User stats not found"}

        current_user.stats.xp += 100
        current_user.stats.coins += 50

        notification_service.create_notification(
        db, 
        current_user.id, 
        "Practice Complete!", 
        f"Earned Practice Rewards: +100 XP, +50 Coins!", 
        "success"
    )
        
        db.add(current_user.stats) 
        db.commit()
        db.refresh(current_user)
        
        return {"success": True, "message": "Practice reward added!"}
    except Exception as e:
        db.rollback()
        return {"success": False, "message": str(e)}