"""
Unified evaluation utilities
"""
from typing import Dict, Any
import numpy as np


def format_metrics(metrics: Dict[str, Any], problem_type: str) -> Dict[str, Any]:
    """Clean metrics for JSON response"""
    clean = {}
    for k, v in metrics.items():
        if k in ("confusion_matrix", "report", "actual", "predicted", "projection"):
            clean[k] = v
        elif isinstance(v, (float, np.floating)):
            clean[k] = round(float(v), 4)
        elif isinstance(v, (int, np.integer)):
            clean[k] = int(v)
        else:
            clean[k] = v
    return clean
