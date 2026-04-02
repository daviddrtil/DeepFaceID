import asyncio
import base64
import json
import ssl
import threading
import time
import webbrowser
import cv2
import numpy as np
from pathlib import Path
from queue import Empty
from aiohttp import web
import settings
from preprocessing.video_input import VideoInput, EndOfStreamError
from preprocessing.live_video_queue import LiveVideoQueue


class WebSocketInput(VideoInput):
    def __init__(self):
        super().__init__()
        self.width = 1920
        self.height = 1080
        self.fps = 30.0
        self.is_live = True
        self.queue = LiveVideoQueue(maxsize=4)
        self.frame_count = 0
        self.start_time = None
        self.connected = threading.Event()

    def print_video_info(self):
        print(f"WebSocket Input: {self.width}x{self.height} @ {self.fps:.0f} fps")

    def put_frame(self, frame, width, height):
        if self.start_time is None:
            self.start_time = time.time()
        self.width = width
        self.height = height
        timestamp_ms = self.frame_count * 100
        self.queue.put_latest((frame, timestamp_ms, self.frame_count))
        self.frame_count += 1

    def get_frame(self):
        try:
            data = self.queue.get(timeout=0.5)
        except Empty:
            if self.stop_event.is_set():
                raise EndOfStreamError()
            raise Empty
        if data is None:
            raise EndOfStreamError()
        return data

    def reset(self):
        self.frame_count = 0
        self.start_time = None
        self.stop_event.clear()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break


class WebOutput:
    def __init__(self):
        self.ws = None
        self.loop = None

    def set_connection(self, ws, loop):
        self.ws = ws
        self.loop = loop

    def start(self):
        pass

    def put_frame(self, frame, frame_count, action_message=None):
        pass

    def put_overlay(self, overlay_data):
        if self.ws and self.loop:
            asyncio.run_coroutine_threadsafe(self._send_overlay(overlay_data), self.loop)

    async def _send_overlay(self, overlay_data):
        if self.ws and not self.ws.closed:
            try:
                await self.ws.send_json(overlay_data)
            except Exception:
                pass

    def stop(self):
        pass


class WebServer:
    def __init__(self, stop_event, video_input, web_output, engine_factory):
        self.stop_event = stop_event
        self.video_input = video_input
        self.web_output = web_output
        self.engine_factory = engine_factory
        self.static_dir = Path(__file__).parent / "static"
        self.sounds_dir = Path(__file__).parent / "sounds"

        self.app = web.Application()
        self._setup_routes()
        self.server_thread = None
        self.loop = None
        self.engine = None
        self.engine_thread = None

    def _setup_routes(self):
        self.app.router.add_get('/', self._serve_index)
        self.app.router.add_static('/static', self.static_dir)
        self.app.router.add_static('/sounds', self.sounds_dir)
        self.app.router.add_get('/ws', self._websocket_handler)

    async def _serve_index(self, request):
        return web.FileResponse(self.static_dir / 'index.html')

    async def _websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.web_output.set_connection(ws, self.loop)
        self._start_engine()

        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)

                if data.get('type') == 'reset':
                    self._stop_engine()
                    self.video_input.reset()
                    self._start_engine()
                    await ws.send_json({'type': 'reset_ack'})
                    continue

                if data.get('type') == 'frame':
                    frame = self._decode_frame(data.get('frame', ''))
                    if frame is not None:
                        self.video_input.put_frame(frame, data.get('width', 640), data.get('height', 480))

            elif msg.type == web.WSMsgType.ERROR:
                break

        self._stop_engine()
        self.web_output.set_connection(None, None)
        return ws

    def _start_engine(self):
        self.engine = self.engine_factory()
        self.engine_thread = threading.Thread(target=self.engine.run, daemon=True)
        self.engine_thread.start()

    def _stop_engine(self):
        if self.engine:
            self.engine.stop_event.set()
        self.video_input.stop()
        if self.engine_thread and self.engine_thread.is_alive():
            self.engine_thread.join(timeout=2.0)
        self.video_input.reset()

    def _decode_frame(self, frame_b64):
        try:
            img_data = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def _run_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        runner = web.AppRunner(self.app)
        self.loop.run_until_complete(runner.setup())

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain("cert.pem", "key.pem")

        site = web.TCPSite(
            runner,
            settings.config.web_host,
            settings.config.web_port,
            ssl_context=ssl_context
        )

        self.loop.run_until_complete(site.start())
        self.loop.run_forever()

    def start(self):
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        url = f"https://{settings.config.web_host}:{settings.config.web_port}"
        print(f"Web server started on URL: {url}")

        def open_browser():
            time.sleep(1)
            webbrowser.open(url)
        threading.Thread(target=open_browser, daemon=True).start()

    def stop(self):
        self._stop_engine()
        self.stop_event.set()
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        print("Web server stopped.")
