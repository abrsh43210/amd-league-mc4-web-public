import os
from pathlib import Path

WORK_DIR = Path(os.environ.get("MC4_WORK_DIR", "/tmp/mc4"))
LOG_FILE = WORK_DIR / "app.log"
SERVER_LOG = WORK_DIR / "server.log"
SOCKET = Path(os.environ.get("MC4_SOCKET", "/tmp/mc4-model.sock"))
OUTPUT_DIR = Path(os.environ.get("MC4_OUTPUT_DIR", "/app/output"))
MODEL_DIR = os.environ.get("MC4_MODEL_DIR", "/models/llm")
DEADLINE_S = float(os.environ.get("MC4_DEADLINE_S", "25"))   # per question, assumed 30 s budget
MAX_PAGES = int(os.environ.get("MC4_MAX_PAGES", "8"))


def log(msg: str) -> None:
    """Logs go to a file, never stdout (stdout noise once cost us OUTPUT_MALFORMED)."""
    import time
    try:
        WORK_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(time.strftime("%H:%M:%S ") + str(msg).rstrip() + "\n")
    except OSError:
        pass
