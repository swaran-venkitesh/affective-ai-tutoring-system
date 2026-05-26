import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import server


def main() -> None:
    server.load_runtime_state()
    mon = server.threading.Thread(target=server._empathy_monitor, daemon=True, name="EmpathyMonitor")
    mon.start()
    night = server.threading.Thread(target=server._night_check, daemon=True, name="NightCheck")
    night.start()
    server.start_background_threads()
    server.socketio.run(server.app, debug=False, port=5001, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
