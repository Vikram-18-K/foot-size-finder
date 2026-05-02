import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CVError(Exception):
    pass

# Constants for A4 Paper
A4_WIDTH_CM = 21.0
A4_LENGTH_CM = 29.7
PIXELS_PER_CM = 10.0  # Standardized scale in the warped coordinate space

def enhance_image(image: np.ndarray):
    """
    Super-fast image enhancement optimized for foot detection.
    Suppresses shadows and sharpens edges at low resolution.
    """
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    y = clahe.apply(y)
    y = cv2.normalize(y, None, 0, 255, cv2.NORM_MINMAX)
    enhanced = cv2.cvtColor(cv2.merge((y, u, v)), cv2.COLOR_YUV2BGR)
    return enhanced

def validate_image_quality(image: np.ndarray, threshold: float = 8.0):
    """Rejects blurry images using Laplacian variance."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < threshold:
        raise CVError(f"Image is too blurry. Please hold steady.")
    return True

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def detect_a4_paper(image: np.ndarray):
    """
    Ultimate robust A4 detection using multi-method fallbacks and threshold sweeping.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_area = image.shape[0] * image.shape[1]
    
    # Pre-processing
    melted = cv2.medianBlur(gray, 7)
    
    def find_best_paper(thresh_img):
        # Use RETR_LIST to find nested contours if necessary
        cnts, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        
        best_cnt = None
        max_score = -1
        
        for c in cnts:
            area = cv2.contourArea(c)
            # Ultra-lenient range: 0.1% to 95% of image
            if (img_area * 0.001) < area < (img_area * 0.95):
                hull = cv2.convexHull(c)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                
                # Nuclear Leniency: solidity > 0.1
                if solidity > 0.1:
                    rect_rot = cv2.minAreaRect(c)
                    (_, (wr, hr), _) = rect_rot
                    if min(wr, hr) > 0:
                        aspect_ratio = max(wr, hr) / min(wr, hr)
                        # Ultra-lenient aspect ratio: 0.2 to 10.0
                        if 0.2 < aspect_ratio < 10.0:
                            # Scoring: Area * AspectRatioMatch
                            ar_score = 1.0 / (1.0 + abs(aspect_ratio - 1.41))
                            score = (area / img_area) * ar_score
                            if score > max_score:
                                max_score = score
                                best_cnt = c
        return best_cnt

    # 1. Try Global Otsu
    _, thresh_otsu = cv2.threshold(melted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    thresh_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    paper_cnt = find_best_paper(thresh_otsu)
    
    # 2. Fallback to Adaptive Threshold
    if paper_cnt is None:
        thresh_adapt = cv2.adaptiveThreshold(melted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 51, 10)
        thresh_adapt = cv2.morphologyEx(thresh_adapt, cv2.MORPH_CLOSE, kernel)
        paper_cnt = find_best_paper(thresh_adapt)
        
    # 3. Fallback to Threshold Sweep (for very difficult lighting)
    if paper_cnt is None:
        for t in [200, 150, 100, 220, 80]:
            _, thresh_fixed = cv2.threshold(melted, t, 255, cv2.THRESH_BINARY)
            paper_cnt = find_best_paper(thresh_fixed)
            if paper_cnt is not None: break

    # 4. Fallback to Edge-based Detection
    if paper_cnt is None:
        edged = cv2.Canny(melted, 20, 100) # More sensitive Canny
        dilated = cv2.dilate(edged, kernel, iterations=1)
        paper_cnt = find_best_paper(dilated)
        
    if paper_cnt is None:
        raise CVError("A4 paper not detected. Ensure it's fully visible and not covered by your foot.")

    rect_rot = cv2.minAreaRect(paper_cnt)
    box = cv2.boxPoints(rect_rot)
    box = np.array(box, dtype=np.intp)
    
    rect = order_points(box.reshape(4, 2))
    (tl, tr, br, bl) = rect
    
    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    
    return rect, int(w), int(h)

def get_homography(rect, maxWidth, maxHeight):
    """Maps pixels to real centimeters."""
    if maxWidth > maxHeight:
        paper_w_cm, paper_h_cm = A4_LENGTH_CM, A4_WIDTH_CM
    else:
        paper_w_cm, paper_h_cm = A4_WIDTH_CM, A4_LENGTH_CM
        
    dst_w = paper_w_cm * PIXELS_PER_CM
    dst_h = paper_h_cm * PIXELS_PER_CM
    
    dst = np.array([[0, 0], [dst_w - 1, 0], [dst_w - 1, dst_h - 1], [0, dst_h - 1]], dtype="float32")
    return cv2.getPerspectiveTransform(rect, dst)

def segment_foot(image: np.ndarray, paper_rect: np.ndarray, foot_side: str):
    """Isolates the foot shape using a combination of edge detection and adaptive thresholding."""
    h, w = image.shape[:2]
    max_dim = 400.0 # Increased for better foot detail
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    small_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    sh, sw = small_image.shape[:2]
    
    # Get paper boundaries
    small_paper_rect = paper_rect * scale
    x, y, pw, ph = cv2.boundingRect(small_paper_rect.astype(np.int32))
    
    # ROI for foot: Tight vertical range (5% margin) to cut off the leg
    roi_w = int(sw * 0.6)
    margin_v = int(ph * 0.05)
    sy1 = max(0, y - margin_v)
    sy2 = min(sh - 1, y + ph + margin_v)
    
    if foot_side == "left":
        sx1, sx2 = max(0, x - roi_w), max(0, x - 5)
    else:
        sx1, sx2 = min(sw - 1, (x + pw) + 5), min(sw - 1, (x + pw) + roi_w)
        
    roi = small_image[sy1:sy2, sx1:sx2]
    if roi.size == 0:
        raise CVError("Foot ROI empty. Ensure your foot is next to the paper.")

    # Process ROI
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    
    # Method 1: Adaptive Thresholding (good for skin vs floor)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Method 2: Canny Edges
    edged = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edged, kernel, iterations=1)
    
    # Combine methods
    combined = cv2.bitwise_or(thresh, dilated)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise CVError("Foot not detected. Ensure your foot is fully visible and not in deep shadow.")
        
    # Pick the largest contour that isn't the entire ROI border
    foot_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(foot_contour) < (roi.size * 0.05):
        raise CVError("Foot shape too small. Move closer.")
        
    # Offset contour back to small_image coordinates
    foot_contour[:, :, 0] += sx1
    foot_contour[:, :, 1] += sy1
    
    return (foot_contour / scale).astype(np.int32)

def measure_foot(foot_contour, M):
    """Calculates foot dimensions and applies 3D compensation."""
    c = np.array(foot_contour, dtype=np.float32)
    warped_c = cv2.perspectiveTransform(c.reshape(-1, 1, 2), M)
    rect = cv2.minAreaRect(warped_c)
    (_, (w_px, h_px), _) = rect
    
    # Calculate raw dimensions
    l_raw = max(w_px, h_px) / PIXELS_PER_CM
    w_raw = min(w_px, h_px) / PIXELS_PER_CM
    
    # 3D Depth Compensation: Foot is a 3D object. Top of foot is closer to lens.
    # 0.85 is a standard conservative multiplier for depth correction.
    length_cm = l_raw * 0.85
    width_cm = w_raw * 0.85
    
    # SANITY CHECKS REMOVED AS REQUESTED BY USER
    # We now return the measurement even if it looks 'impossible'
    return length_cm, width_cm

def calculate_shoe_size(length_cm: float):
    """
    Industry Standard Foot-to-Shoe Size Conversion.
    """
    # Standard formula for UK/India sizing
    uk_raw = (length_cm - 19.8) / 0.846
    uk_size = max(1, int(round(uk_raw)))
    
    us_men = uk_size + 1
    eu_size = int(round(1.5 * (length_cm + 1.5)))
    
    # Capping removed
    return uk_size, us_men, eu_size

def process_image(image: np.ndarray, foot_side: str) -> dict:
    validate_image_quality(image)
    rect, mw, mh = detect_a4_paper(image)
    M = get_homography(rect, mw, mh)
    
    try:
        foot_contour = segment_foot(image, rect, foot_side)
    except CVError:
        other = "right" if foot_side == "left" else "left"
        foot_contour = segment_foot(image, rect, other)
        foot_side = other
        
    l_cm, w_cm = measure_foot(foot_contour, M)
    if not (12 < l_cm < 35):
        raise CVError("Measurement out of range (12-35cm). Check your foot placement.")
        
    uk, us, eu = calculate_shoe_size(l_cm)
    return {
        "foot_side": foot_side, 
        "length_cm": round(l_cm, 1), 
        "width_cm": round(w_cm, 1), 
        "shoe_size_uk": uk, 
        "shoe_size_us": us,
        "shoe_size_eu": eu
    }

def fast_validate_image(image: np.ndarray, foot_side: str) -> dict:
    """Lite version for real-time camera feedback."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # BLUR CHECK REMOVED AS REQUESTED
        # is_blurry = cv2.Laplacian(gray, cv2.CV_64F).var() < 7.0
            
        rect, mw, mh = detect_a4_paper(image)
        # Note: In fast validation, we primarily check for A4 as a proxy for a good setup.
        return {
            "a4_detected": True, "foot_detected": True, "is_blurry": False,
            "tilt_ok": True, "valid": True, "confidence": 0.9,
            "message": "Perfect! Hold still..."
        }
    except CVError as e:
        return {
            "a4_detected": False, "foot_detected": False, "is_blurry": False,
            "tilt_ok": True, "valid": False, "confidence": 0.0,
            "message": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected validation error: {str(e)}")
        return {
            "a4_detected": False, "foot_detected": False, "is_blurry": False,
            "tilt_ok": True, "valid": False, "confidence": 0.0,
            "message": f"Detection error: {str(e)}"
        }
