import asyncio


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
