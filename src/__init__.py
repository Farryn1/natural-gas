"""Henry Hub natural gas price forecasting package."""
import os

# Allow multiple OpenMP runtimes to coexist. xgboost, scikit-learn, and torch each
# ship their own libomp; on macOS loading several segfaults without this flag.
# Must be set before any of those libraries are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
