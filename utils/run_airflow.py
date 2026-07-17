import os
import sys

# 1. Add project root to python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set Airflow environment variables dynamically
os.environ['AIRFLOW__CORE__DAGS_FOLDER'] = os.path.join(PROJECT_ROOT, 'dags')
os.environ['AIRFLOW__CORE__LOAD_EXAMPLES'] = 'False'

# 2. Prevent ThreadPoolExecutor import issues during dynamic provider loading
from concurrent.futures import ThreadPoolExecutor

# 3. Mock os.register_at_fork (missing on Windows)
if not hasattr(os, 'register_at_fork'):
    os.register_at_fork = lambda *args, **kwargs: None

# 4. Mock fcntl, pwd, and resource modules (missing on Windows)
from unittest.mock import MagicMock
fcntl_mock = MagicMock()
fcntl_mock.LOCK_SH = 1
fcntl_mock.LOCK_EX = 2
fcntl_mock.LOCK_NB = 4
fcntl_mock.LOCK_UN = 8
fcntl_mock.flock = lambda fd, op: None
sys.modules['fcntl'] = fcntl_mock
sys.modules['pwd'] = MagicMock()
resource_mock = MagicMock()
resource_mock.RLIMIT_NOFILE = 7
resource_mock.getrlimit = lambda x: (1024, 1024)
sys.modules['resource'] = resource_mock

# 5. Mock Unix-specific signals on Windows
import signal
# Airflow's scheduler registers these POSIX-only signals.  Define placeholders
# on Windows; _mocked_signal below safely ignores their registration.
for _name, _value in {
    'SIGQUIT': 3,
    'SIGUSR1': 10,
    'SIGUSR2': 12,
}.items():
    if not hasattr(signal, _name):
        setattr(signal, _name, _value)

_original_signal = signal.signal
def _mocked_signal(sig, handler):
    try:
        return _original_signal(sig, handler)
    except (ValueError, OSError):
        # Ignore POSIX-only signal registrations on Windows.
        return None

signal.signal = _mocked_signal

# 6. Monkeypatch Pydantic FieldInfo._copy (removed in Pydantic 2.10+)
import copy
from pydantic.fields import FieldInfo
if not hasattr(FieldInfo, '_copy'):
    def _copy_field_info(self, **kwargs):
        c = copy.copy(self)
        for k, v in kwargs.items():
            setattr(c, k, v)
        return c
    FieldInfo._copy = _copy_field_info

# 7. Run Airflow entrypoint
from airflow.__main__ import main

if __name__ == '__main__':
    main()
