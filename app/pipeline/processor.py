from ..defs import PreprocessParams, DetectParams
from ..state import ImageState

import cv2
import numpy as np
import matplotlib.pyplot as plt


try:
    from skimage.feature import local_binary_pattern
    _HAS_SKIMG = True
except:
    _HAS_SKIMG = False


class Processor:
    def __init__(self):
        self.preprocess_params = PreprocessParams()
        self.detect_params = DetectParams()
    

    def preprocess(self, img_st: ImageState): 
        img = img_st.original
        if img is None: return

        img = self._scale_img(img)

        if img_st.custom:
            method = img_st.preprocess_params.gray_method
        else:
            method = self._auto_grayscale_stable(img)
            
        gray = self._to_grayscale(img, method)

        if img_st.preprocess_params.use_clahe:
            grid = img_st.preprocess_params.clahe_grid

            gray = self._apply_clahe(
                gray, 
                img_st.preprocess_params.clahe_clip,
                (grid, grid)
            )
        
        texture = self._estimate_texture_level(gray)
        img_st.info = ""

        return gray, texture
        

    def detect(self, img_st: ImageState):
        if img_st.preprocessed is None or img_st.preprocessed.img is None:
            return None
        
        if img_st.custom:
            method = img_st.detect_params.method
            img_st.info = "Manual: " + str(method)
            return self._dispatch_manual_detect(img_st, method)
    
        img_st.info = "Auto: variance-core"
        return self._detect_variance_core(img_st)
    

    def show_variance_histogram(self, img_st: ImageState):
        gray = img_st.preprocessed.img
        if gray is None: return

        var_map = self._variance_multiscale(gray, scales=self._get_scales(img_st))
        th = self._compute_threshold(var_map, img_st.detect_params)
        self._plot_histogram(var_map, th)

    
    def _apply_clahe(self, gray: np.ndarray, clip=2.0, tile_grid=(8, 8)):
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tuple(tile_grid))
        return clahe.apply(gray)
    

    def _compute_threshold(self, var_map_u8: np.ndarray, params: DetectParams):
        if params.th_mode == "fixed":
            return params.fixed_th

        if params.th_mode == "zscore":
            x = var_map_u8.astype(np.float32)
            med = np.median(x)
            mad = np.median(np.abs(x - med)) + 1e-6
            # 1.4826 * MAD approximates std for normal distribution
            robust_std = 1.4826 * mad
            return float(med + params.z_k * robust_std)

        p = max(50.0, min(99.5, params.percentile))

        return float(np.percentile(var_map_u8, p))
    

    def _dispatch_manual_detect(self, st: ImageState, m: str):
        if m == "variance" or m == "var_lbp":
            return self._detect_variance_core(st)

        if m == "adaptive":
            return self._detect_adaptive(st)

        if m == "edge":
            return self._detect_edge_density(st)
        
        if m == "saturation":
            return self._detect_saturation(st)
        
        raise ValueError(f"Unknown detection method: {m}")
    

    def _auto_grayscale_stable(self, bgr: np.ndarray):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]
        mean_s = float(np.mean(s))

        # clearly colored surface, often preserves contrast better
        if mean_s > 60: return "weighted"

        # stable, perceptual
        return "luminosity"
    

    def _to_grayscale(self, img, method="weighted"):  
        b, g, r = cv2.split(img)

        if method == "weighted":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if method == "average":
            return ((r + g + b) / 3).astype(np.uint8)
        
        if method == "max":
            return np.maximum(np.maximum(r, g), b)
        
        if method == "min":
            return np.minimum(np.minimum(r, g), b)
        
        if method == "luminosity":
            return (0.21 * r + 0.72 * g + 0.07 * b).astype(np.uint8)
        
        raise ValueError(f"Unknown method: {method}")
    

    def _detect_variance_core(self, st: ImageState):
        gray = st.preprocessed.img
        if gray is None: return None

        params = st.detect_params

        # multi-scale var map
        scales = self._get_scales(st)
        var_map = self._variance_multiscale(gray, scales=scales)

        # robust th.ing on var map
        th = self._compute_threshold(var_map, params)

        mask = (var_map > th).astype(np.uint8) * 255
        mask = self._filter_components_by_area(mask, params.min_area, params.max_area)

        # optional -- LBP-uniformity filter (candidate validation)
        if params.use_lbp:
            mask = self._refine_with_lbp(gray, mask, params)

        # morphology
        mask = self._morph_refine(mask, params.elemsize, params.open_iter, params.close_iter)

        return self._apply_mask(st.original, mask)
    

    def _detect_adaptive(self, img_st: ImageState):
        gray = img_st.preprocessed.img
        if gray is None: return

        # consider enabling clahe when using this
        block = img_st.detect_params.block_size
        if block < 3:
            block = 31
        if block % 2 == 0:
            block += 1

        mask = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=block,
            C=img_st.detect_params.c
        )
        
        mask = self._morph_refine(mask, img_st.detect_params.elemsize, open_iter=1, close_iter=1)

        return self._apply_mask(img_st.original, mask)


    def _detect_edge_density(self, img_st: ImageState):
        gray = img_st.preprocessed.img
        if gray is None: return

        edges = cv2.Canny(gray, img_st.detect_params.edge_t1, img_st.detect_params.edge_t2)

        k = img_st.detect_params.edge_kernel
        if k < 3:
            k = 9
        if k % 2 == 0:
            k += 1

        kernel = np.ones((k, k), np.uint8)
        density = cv2.filter2D(edges.astype(np.float32), -1, kernel)

        th = img_st.detect_params.edge_density_th
        _, mask = cv2.threshold(density, th, 255, cv2.THRESH_BINARY)

        mask = mask.astype(np.uint8)
        mask = self._morph_refine(mask, img_st.detect_params.elemsize, open_iter=1, close_iter=1)

        return self._apply_mask(img_st.original, mask)


    def _detect_saturation(self, img_st: ImageState):
        img = img_st.original
        if img is None: return

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        s = hsv[:, :, 1]

        th = img_st.detect_params.edge_density_th
        _, mask = cv2.threshold(s, th, 255, cv2.THRESH_BINARY_INV)

        mask = self._morph_refine(mask, img_st.detect_params.elemsize, open_iter=1, close_iter=1)

        return self._apply_mask(img_st.original, mask)
    

    def _get_scales(self, st: ImageState):
        scales = st.detect_params.scales
        if scales and isinstance(scales, (list, tuple)) and len(scales) > 0:
            # ensure odd and >=3
            out = []
            for k in scales:
                k = int(k)
                if k < 3:
                    k = 3
                if k % 2 == 0:
                    k += 1
                out.append(k)
            return out

        txt = st.preprocessed.texture

        if txt == "low_txt":
            return [5, 9, 13]
        
        if txt == "mid_txt":
            return [7, 11, 15]
        
        return [9, 15, 21]
    

    def _variance_multiscale(self, gray: np.ndarray, scales, normalize=True):
        gray_f = gray.astype(np.float32)
        combined = None

        for k in scales:
            mean = cv2.boxFilter(gray_f, ddepth=-1, ksize=(k, k), normalize=True)
            sq_mean = cv2.boxFilter(gray_f * gray_f, ddepth=-1, ksize=(k, k), normalize=True)
            var = sq_mean - mean * mean
            if combined is None:
                combined = var
            else:
                combined = np.maximum(combined, var)

        if not normalize: return combined

        hi = np.percentile(combined, 99.5)
        lo = np.percentile(combined, 1.0)
        clipped = np.clip(combined, lo, hi)

        norm = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)
    

    def _filter_components_by_area(self, mask: np.ndarray, min_ratio: float, max_ratio: float):
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        h, w = mask.shape[:2]
        img_area = float(h * w)

        out = np.zeros_like(mask)
        for i in range(1, num):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < min_ratio * img_area:
                continue
            if area > max_ratio * img_area:
                continue
            out[labels == i] = 255

        return out
    

    def _refine_with_lbp(self, gray: np.ndarray, mask: np.ndarray, params: DetectParams):
        if not _HAS_SKIMG: return mask
        
        lbp = local_binary_pattern(gray, params.lbp_points, params.lbp_rad, method="uniform")

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)

        for i in range(1, num):
            comp = (labels == i)
            region = lbp[comp]
            if region.size == 0:
                continue

            # In 'uniform' LBP, uniform patterns are <= p, non-uniform are > p
            uniform_ratio = float(np.sum(region <= params.lbp_points)) / float(region.size)

            # too uniform -> likely wall texture/paint, not mold
            if uniform_ratio > params.lbp_uniform_th:
                continue

            out[comp] = 255

        return out
    

    def _morph_refine(self, mask: np.ndarray, elemsize=7, open_iter=1, close_iter=1):
        elemsize = max(3, int(elemsize))
        if elemsize % 2 == 0:
            elemsize += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (elemsize, elemsize))
        if close_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        if open_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        
        return mask


    def _estimate_texture_level(self, gray: np.ndarray):
        var = self._variance_multiscale(gray, scales=[9], normalize=True)
        tail = float(np.mean(var > 60))

        if tail < 0.01:
            return "low_txt"
        if tail < 0.04:
            return "mid_txt"
        return "high_txt"
    

    def _scale_img(self, img, max_dim=1024):
        h,w = img.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img    

    
    def _apply_mask(self, img:np.ndarray, mask=None):
        if img is None: return None

        # resize and overlay
        oh, ow = img.shape[:2]

        if mask is not None:
            mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)

        if img.ndim == 2:
            base_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = img.copy()

        display = base_bgr.copy()

        if mask is not None:
            _red = np.zeros_like(display)
            _red[:, :, 2] = 255
            display = cv2.addWeighted(display, 0.7, _red, 0.3, 0)
            display[mask==0] = base_bgr[mask==0]

        return display
    

    def _plot_histogram(self, var_map, curr_th):
        plt.figure(figsize=(6, 4))
        plt.hist(var_map.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
        plt.axvline(curr_th, color='r', linestyle='--', label=f"th={curr_th:.2f}")
        plt.legend()
        plt.title("Variance Histogram + Threshold")
        plt.xlabel("Variance Value")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()