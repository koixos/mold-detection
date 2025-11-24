import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries
from skimage.graph import rag_mean_color, cut_threshold

def detect_mold_texture(img, th=150, ksize=11, elemsize=5):
    var_map = local_variance(img, ksize)
    var_norm = cv2.normalize(var_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(var_norm, th, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (elemsize, elemsize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return mask, var_norm

def local_variance(gray, ksize=9):
    mean = cv2.blur(gray.astype(np.float32), (ksize, ksize))
    sq_mean = cv2.blur((gray.astype(np.float32) ** 2), (ksize, ksize))
    return sq_mean - mean ** 2

def detect_background_brightness(img, sample_center=False):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]

    if sample_center:
        h, w = v.shape
        ch, cw = h // 2, w // 2
        region = v[ch - h // 6 : ch + h // 6,
                   cw - w // 6 : cw + w // 6]
    else:
        region = v

    mean_val = np.mean(region)

    if mean_val < 85:   
        brightness = "dark"
    elif mean_val < 170:
        brightness = "medium"
    else:
        brightness = "light"

    return brightness, mean_val

def to_grayscale(img, method="weighted"):  
    b, g, r = cv2.split(img)
    if method == "weighted":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif method == "average":
        return ((r + g + b) / 3).astype(np.uint8)
    elif method == "max":
        return np.maximum(np.maximum(r, g), b)
    elif method == "min":
        return np.minimum(np.minimum(r, g), b)
    elif method == "luminosity":
        return (0.21 * r + 0.72 * g + 0.07 * b).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method: {method}")

def execute(path): 
    original_img = load_imgs(path)
    visualize(original_img, title="original")
    
    brightness, _ = detect_background_brightness(original_img)
    
    method = ""
    th = 0
    ksize = 0
    elemsize = 0

    if brightness == "light":
        method = "average"
        th = 75
        ksize = 11
        elemsize = 8
    else:
        return
        method = "max"
        th = 15
        ksize = 11
        elemsize = 8
    
    gray = to_grayscale(original_img, method)
    visualize(gray, title="original")

    mask, var_map = detect_mold_texture(gray, th, ksize, elemsize)
        #cv2.imshow("Texture Variance", var_map)
    visualize(original_img, mask)

    #cv2.imshow("Texture Variance", var_map)
    #cv2.imshow("Mold Mask", mask)


def visualize(img, mask=None, title="Mold Candidates Overlay"):
    display = img.copy()

    if mask is not None:
        _red = np.zeros_like(display)
        _red[:, :, 2] = 255
        display = cv2.addWeighted(display, 0.7, _red, 0.3, 0)
        display[mask==0] = img[mask==0]

    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.title(title)
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.show()


def segmentation(img, n_segments=250, compactness=20):
    # using SLIC algo for segmentation
    img_float = img_as_float(img)
    segments = slic(img_float, n_segments=n_segments, compactness=compactness, start_label=1)
    return segments

def analyze_segments(img, segments, v_thresh=100):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    global_v_mean = np.mean(hsv[:, :, 2])
    global_s_mean = np.mean(hsv[:, :, 1])
    
    for seg_val in np.unique(segments):
        seg_mask = (segments == seg_val)
        
        v_mean =  np.mean(hsv[:, :, 2][seg_mask])
        
        s_mean = np.mean(hsv[:, :, 1][seg_mask])
        s_std = np.std(hsv[:, :, 1][seg_mask])

        b_mean = np.mean(lab[:, :, 2][seg_mask])   

        if (
            (v_mean < global_v_mean - 10 and s_mean > global_s_mean) or
            (s_std > 20 and v_mean < 130) or
            (b_mean > 135)
        ):
            mask[seg_mask] = 255

    return mask

def merge_similar_segments(img, segments, th=25):
    g = rag_mean_color(img, segments)
    merged = cut_threshold(segments, g, th)
    return merged

def refine_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask

def run(path):
    original_img = load_imgs(path)
    #visualize_img(original_img)

    processed_img = preprocess(original_img)
    #visualize_img(processed_img)

    slic_segments = segmentation(processed_img)
    #visualize_segments(original_img, slic_segments)

    merged_segments = merge_similar_segments(processed_img, slic_segments)
    #visualize_segments(original_img, merged_segments)

    #mold_mask = detect_mold(processed_img)

    mold_mask = analyze_segments(processed_img, merged_segments)
    mold_mask = refine_mask(mold_mask)

    visualize_img(original_img, mold_mask)

def preprocess(img):
    """
    2) IMAGE ENHANCEMENT
    """
    img_smooth = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    #img_norm = cv2.normalize(img_smooth, None, 0, 255, cv2.NORM_MINMAX)
    return img_smooth.astype(np.uint8)

def detect_mold(img):
    """
    3) COLOR-BASED ADAPTIVE SEGMENTATION
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)        # hue, saturation, value/lightness

    local_mean = cv2.blur(v, (15, 15))    # adaptive local mean

    mean_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if mean_brightness > 128:
        mask1 = (v < (local_mean - 5)) & (s > 40)
    else:
        mask1 = (v > (local_mean + 3)) & (s > 40)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)        # lightness, green->red, blue->yellow

    mask2 = (b > 135)                # blue2yellow > 135 ise sarı-kahverengi tonlara daha yakın

    # düşük parlaklık, yüksek doygunluk, b kanalı yüksek
    mask = (mask1 | mask2).astype('uint8') * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    # maskedeki delikleri (küf alanı içindeki siyah boşluklar) kapat
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # kapandıktan sonra etrafta kalan küçük, izole gürültüleri siler
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # final temizlik - alan tabanlı temizlik
    mask = remove_small_objs(mask, min_size=1000)
    
    return mask

def visualize_img(img, mask=None):
    display = img.copy()
    title = ""

    if mask is not None:
        _red = np.zeros_like(display)
        _red[:, :, 2] = 255
        display = cv2.addWeighted(display, 0.7, _red, 0.3, 0)
        display[mask==0] = img[mask==0]
        title = "Mold Candidates Overlay"

    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.title(title)
    plt.imshow(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    plt.show()

def visualize_segments(img, segments):
    # visualize the SLIC segments on the original image
    fig, ax = plt.subplots(figsize=(10, 6))
    boundaries = mark_boundaries(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), segments)
    ax.imshow(boundaries)
    ax.set_title("SLIC Segments")
    ax.axis('off')
    plt.show()

def remove_small_objs(mask, min_size=500):
    # min_size pixelden az pixel içeren küçük alanları temizle
    bw = (mask > 0).astype(np.uint8)
    labels = measure.label(bw, connectivity=2)
    out = np.zeros_like(bw)
    for region in measure.regionprops(labels):
        if region.area >= min_size:
            out[labels == region.label] = 1
    return (out * 255).astype('uint8')

def load_imgs(path, max_dim=1024):
    """
    1) IMAGE ACQUISITION & SCALING
        görseli yüklüyor ve eğer görselin h veya w boyutlarından en az biri
        max_dim ile belirtilen pixel boyutundan büyükse en büyük olan boyutu
        max_dim'e eşitleyecek şekilde iki boyutu da resize ediyor ve son
        görseli dönüyor
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    h,w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img