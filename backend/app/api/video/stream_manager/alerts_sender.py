import requests
import queue
from threading import Thread


class AlertsSender:
    def __init__(self):
        self.queue = queue.Queue(maxsize=100)
        self.running = True
        self.thread = Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.error_count = 0
        self.success_count = 0
    
    def _worker(self):
        while self.running:
            try:
                payload = self.queue.get(timeout=1.0)
                if payload is None:
                    break
                try:
                    if payload.get('alert_type') == 'behavior':
                        response = requests.post(
                            "http://127.0.0.1:5000/api/alerts/create",
                            json=payload,
                            timeout=2.0
                        )
                    else:
                        response = requests.post(
                            "http://127.0.0.1:5000/api/alerts/yolo-detection",
                            json=payload,
                            timeout=2.0
                        )
                    if response.status_code in [200, 201]:
                        self.success_count += 1
                    else:
                        self.error_count += 1
                        if self.error_count % 10 == 0:
                            print(f"[AlertsSender] ⚠️ HTTP {response.status_code}: {response.text[:100]}")
                except requests.exceptions.ConnectionError:
                    self.error_count += 1
                    if self.error_count == 1:
                        print(f"[AlertsSender] ⚠️ Cannot connect to alerts API. Alerts will be skipped.")
                except requests.exceptions.Timeout:
                    self.error_count += 1
                except Exception as e:
                    self.error_count += 1
                    if self.error_count % 20 == 0:
                        print(f"[AlertsSender] ⚠️ Error: {e}")
                self.queue.task_done()
            except queue.Empty:
                continue
    
    def send(self, payload):
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            pass
    
    def get_stats(self):
        return {
            'success': self.success_count,
            'errors': self.error_count,
            'queue_size': self.queue.qsize()
        }
    
    def stop(self):
        self.running = False
        self.queue.put(None)
        self.thread.join(timeout=2.0)