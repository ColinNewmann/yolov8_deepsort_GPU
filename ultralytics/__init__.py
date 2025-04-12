# Ultralytics YOLO 🚀, GPL-3.0 license
print("👀 你正在使用 Colin 改過的 ultralytics")
__version__ = "8.0.3"

from ultralytics.hub import checks
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.utils import ops

__all__ = ["__version__", "YOLO", "hub", "checks"]  # allow simpler import
