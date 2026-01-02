"""
Web Dashboard for Dogfight Visualization

Browser-based 3D visualization using Three.js.
Connects via WebSocket bridge to Python ZeroMQ stream.
"""

from .server import WebDashboardServer, start_web_dashboard

__all__ = [
    "WebDashboardServer",
    "start_web_dashboard",
]
