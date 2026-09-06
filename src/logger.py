import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from src.safety import anonymize

class AuditLogger:
    def __init__(self, path="logs/audit.jsonl"):
        self.path = Path(path)
        self.lock = threading.Lock()

    def log(self, request_id, event, **details):
        record = {"timestamp": datetime.now(timezone.utc).isoformat(),
                  "request_id": request_id, "event": event, **details}
        # Remove obvious identifiers even in nested strings; only synthetic cases supported.
        def redact(value):
            if isinstance(value, str): return anonymize(value)
            if isinstance(value, dict): return {k: redact(v) for k, v in value.items()}
            if isinstance(value, list): return [redact(v) for v in value]
            return value
        line = json.dumps(redact(record), ensure_ascii=False)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
