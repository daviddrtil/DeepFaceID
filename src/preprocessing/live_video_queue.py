import queue

class LiveVideoQueue(queue.Queue):
    def put_latest(self, item):
        while True:
            try:
                self.put(item, block=False)
                break
            except queue.Full:
                try:
                    self.get_nowait()
                except queue.Empty:
                    pass
