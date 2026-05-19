"""Per-model inference adapters. Each adapter normalizes its output to the
shared :class:`ensemble.adapters.base.Prediction` contract so the pipeline
can fuse and evaluate them uniformly.
"""

from ensemble.adapters.base import Adapter, Prediction
from ensemble.adapters.deimv2_adapter import DEIMv2Adapter
from ensemble.adapters.rfdetr_adapter import RFDETRAdapter
from ensemble.adapters.yolo_adapter import YOLOAdapter

__all__ = ["Adapter", "Prediction", "RFDETRAdapter", "YOLOAdapter", "DEIMv2Adapter"]
