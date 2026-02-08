from dataclasses import dataclass
import numpy as np


EXTS = (
    ".png", 
    ".jpg", 
    ".jpeg"
)

PREPROCESS_METHODS = [
    "weighted",
    "average",
    "max",
    "min",
    "luminosity",
]

DETECT_METHODS = [
    "variance",
    "adaptive",
    "edge",
    "saturation",
    "var_lbp",
]

TH_MODES = [
    "percentile",
    "zscore",
    "fixed",
]


@dataclass
class PreprocessedImage:
    img: np.ndarray | None = None
    texture: str = "low_txt"


@dataclass
class PreprocessParams:
    gray_method: str = PREPROCESS_METHODS[0]
    use_clahe: bool = False
    clahe_clip: float = 2.0
    clahe_grid: int = 8


@dataclass
class DetectParams:
    method: str = DETECT_METHODS[0]
    th_mode: str = TH_MODES[0]
    fixed_th: int = 120
    z_k: float = 3.0
    percentile: float = 85.0
    use_lbp: bool = False
    lbp_rad: int = 2
    lbp_points: int = 16
    lbp_uniform_th: float = 0.8
    min_area: float = 0.0005
    max_area: float = 0.35
    scales: list | tuple | None = None
    elemsize: int = 7
    open_iter: int = 1
    close_iter: int = 1
    block_size: int = 31
    c: int = 5
    edge_t1: int = 50
    edge_t2: int = 150
    edge_kernel: int = 9
    edge_density_th: int = 20