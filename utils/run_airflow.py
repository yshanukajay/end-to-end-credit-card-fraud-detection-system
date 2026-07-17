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

# 4. Mock fcntl module (missing on Windows)
from unittest.mock import MagicMock
fcntl_mock = MagicMock()
fcntl_mock.LOCK_SH = 1
fcntl_mock.LOCK_EX = 2
fcntl_mock.LOCK_NB = 4
fcntl_mock.LOCK_UN = 8
fcntl_mock.flock = lambda fd, op: None
sys.modules['fcntl'] = fcntl_mock

# 5. Run Airflow entrypoint
from airflow.__main__ import main

if __name__ == '__main__':
    main()
