from typing import List, Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from datetime import datetime

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_data: Dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None):
        """Accept WebSocket connection and add to active connections"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Fix: Handle None client_id properly
        actual_client_id = client_id if client_id is not None else f"client_{len(self.active_connections)}"
        
        self.client_data[websocket] = {
            "client_id": actual_client_id,
            "connected_at": datetime.now(),
            "subscriptions": []  # What events this client wants to receive
        }
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "client_id": actual_client_id,
            "message": "Connected to Motion Detection System"
        }, websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket from active connections"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.client_data:
            del self.client_data[websocket]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending message to client: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, event_type: Optional[str] = None):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                # Check if client is subscribed to this event type
                client_subs = self.client_data.get(connection, {}).get("subscriptions", [])
                if not event_type or not client_subs or event_type in client_subs:
                    await connection.send_text(json.dumps(message))
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_motion_detection(self, zone_name: str, motion_detected: bool, confidence: float = 0.0):
        """Broadcast motion detection event"""
        message = {
            "type": "motion_detection",
            "timestamp": datetime.now().isoformat(),
            "zone": zone_name,
            "motion_detected": motion_detected,
            "confidence": confidence
        }
        await self.broadcast(message, "motion_detection")

    async def broadcast_beam_event(self, action: str, zone: str, message: str):
        """Broadcast beam control event"""
        event = {
            "type": "beam_event",
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "zone": zone,
            "message": message
        }
        await self.broadcast(event, "beam_events")

    async def broadcast_system_status(self, status: dict):
        """Broadcast system status update"""
        message = {
            "type": "system_status",
            "timestamp": datetime.now().isoformat(),
            "data": status
        }
        await self.broadcast(message, "system_status")

# Global connection manager instance
manager = ConnectionManager()
