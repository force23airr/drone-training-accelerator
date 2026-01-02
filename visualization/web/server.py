#!/usr/bin/env python3
"""
Web Dashboard Server

Bridges ZeroMQ state stream to WebSocket for browser clients.
Serves static files (Three.js viewer).

Usage:
    python -m visualization.web.server --zmq-host localhost --zmq-port 5555 --web-port 8080
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from typing import Optional, Set
from pathlib import Path

try:
    import websockets
    from websockets.server import serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from aiohttp import web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# Add project root
sys.path.insert(0, '.')
from visualization.streaming import StateReceiver

logger = logging.getLogger(__name__)


class WebDashboardServer:
    """
    WebSocket server bridging ZeroMQ to browser clients.

    Features:
    - Receives state from ZeroMQ StateStreamer
    - Broadcasts to connected WebSocket clients
    - Serves static files (HTML, JS, CSS)
    """

    def __init__(
        self,
        zmq_host: str = "localhost",
        zmq_port: int = 5555,
        web_port: int = 8080,
        static_dir: Optional[str] = None,
    ):
        """
        Initialize web dashboard server.

        Args:
            zmq_host: ZeroMQ host to connect to
            zmq_port: ZeroMQ port
            web_port: Web server port
            static_dir: Directory for static files
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("websockets required: pip install websockets")
        if not HAS_AIOHTTP:
            raise ImportError("aiohttp required: pip install aiohttp")

        self.zmq_host = zmq_host
        self.zmq_port = zmq_port
        self.web_port = web_port

        # Static files directory
        if static_dir:
            self.static_dir = Path(static_dir)
        else:
            self.static_dir = Path(__file__).parent / "static"

        # State
        self.receiver: Optional[StateReceiver] = None
        self.clients: Set = set()
        self.running = False

        # Latest frame for new clients
        self.latest_frame_json: Optional[str] = None

    async def start(self):
        """Start the web dashboard server."""
        self.running = True

        # Connect to ZeroMQ
        self.receiver = StateReceiver(self.zmq_host, self.zmq_port)
        self.receiver.connect()
        logger.info(f"Connected to ZeroMQ at {self.zmq_host}:{self.zmq_port}")

        # Start HTTP server for static files
        app = web.Application()
        app.router.add_get('/', self._handle_index)
        app.router.add_static('/static/', self.static_dir)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.web_port)
        await site.start()
        logger.info(f"HTTP server started at http://localhost:{self.web_port}")

        # Start WebSocket server
        ws_port = self.web_port + 1
        async with serve(self._handle_websocket, "0.0.0.0", ws_port):
            logger.info(f"WebSocket server started at ws://localhost:{ws_port}")

            # Start ZeroMQ receiver loop
            await self._zmq_loop()

    async def _handle_index(self, request):
        """Serve index.html."""
        index_path = self.static_dir / "index.html"
        if index_path.exists():
            return web.FileResponse(index_path)
        return web.Response(text="Web Dashboard - index.html not found", status=404)

    async def _handle_websocket(self, websocket):
        """Handle WebSocket client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")

        try:
            # Send latest frame if available
            if self.latest_frame_json:
                await websocket.send(self.latest_frame_json)

            # Keep connection alive
            async for message in websocket:
                # Handle client messages (e.g., camera controls)
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def _handle_client_message(self, websocket, data: dict):
        """Handle message from client."""
        msg_type = data.get("type")

        if msg_type == "request_state":
            # Client requesting current state
            if self.latest_frame_json:
                await websocket.send(self.latest_frame_json)

    async def _zmq_loop(self):
        """Receive from ZeroMQ and broadcast to WebSocket clients."""
        while self.running:
            frame = self.receiver.receive(non_blocking=True)

            if frame:
                # Convert to JSON
                frame_dict = {
                    "type": "frame",
                    "timestamp": frame.timestamp,
                    "match_time": frame.match_time,
                    "red_score": frame.red_score,
                    "blue_score": frame.blue_score,
                    "uavs": [
                        {
                            "id": uav.id,
                            "team": uav.team,
                            "position": list(uav.position),
                            "velocity": list(uav.velocity),
                            "orientation": list(uav.orientation),
                            "speed": uav.speed,
                            "altitude": uav.altitude,
                            "health": uav.health,
                            "alive": uav.alive,
                            "kills": uav.kills,
                            "deaths": uav.deaths,
                            "g_force": uav.g_force,
                            "current_maneuver": uav.current_maneuver,
                        }
                        for uav in frame.uavs
                    ],
                    "events": [
                        {
                            "event_type": event.event_type,
                            "attacker_id": event.attacker_id,
                            "target_id": event.target_id,
                            "weapon": event.weapon,
                            "damage": event.damage,
                            "position": list(event.position),
                        }
                        for event in frame.events
                    ],
                }
                if frame.arena_size is not None:
                    frame_dict["arena_size"] = frame.arena_size
                    frame_dict["arena_height_min"] = frame.arena_height_min
                    frame_dict["arena_height_max"] = frame.arena_height_max

                self.latest_frame_json = json.dumps(frame_dict)

                # Broadcast to all clients
                if self.clients:
                    await asyncio.gather(
                        *[client.send(self.latest_frame_json) for client in self.clients],
                        return_exceptions=True,
                    )

            # Small delay to prevent busy loop
            await asyncio.sleep(1/60)

    def stop(self):
        """Stop the server."""
        self.running = False
        if self.receiver:
            self.receiver.disconnect()


async def start_web_dashboard(
    zmq_host: str = "localhost",
    zmq_port: int = 5555,
    web_port: int = 8080,
):
    """
    Start web dashboard server.

    Args:
        zmq_host: ZeroMQ host
        zmq_port: ZeroMQ port
        web_port: Web server port
    """
    server = WebDashboardServer(zmq_host, zmq_port, web_port)
    await server.start()


def main():
    parser = argparse.ArgumentParser(description='Web Dashboard Server')
    parser.add_argument('--zmq-host', default='localhost', help='ZeroMQ host')
    parser.add_argument('--zmq-port', type=int, default=5555, help='ZeroMQ port')
    parser.add_argument('--web-port', type=int, default=8080, help='Web server port')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("WEB DASHBOARD SERVER")
    print("=" * 60)
    print()
    print(f"ZeroMQ:    {args.zmq_host}:{args.zmq_port}")
    print(f"HTTP:      http://localhost:{args.web_port}")
    print(f"WebSocket: ws://localhost:{args.web_port + 1}")
    print()
    print("Open http://localhost:{} in your browser".format(args.web_port))
    print()

    asyncio.run(start_web_dashboard(args.zmq_host, args.zmq_port, args.web_port))


if __name__ == '__main__':
    main()
