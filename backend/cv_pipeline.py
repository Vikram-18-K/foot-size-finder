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

def validate_image_quality(image: np.ndarray, threshold: float = 12.0):
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
    Robust A4 detection using adaptive thresholding.
    Finds the largest bright 4-sided object in the frame.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use Adaptive Thresholding to find white paper on dark floor
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise CVError("A4 paper not detected. Ensure it's fully visible on a dark floor.")
        
    img_area = image.shape[0] * image.shape[1]
    paper_contour = None
    max_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        if area > (img_area * 0.08): # Paper must be > 8% of image
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and area > max_area:
                paper_contour = approx
                max_area = area
                
    if paper_contour is None:
        raise CVError("A4 paper edges not detected. Check for glare or shadows.")

    rect = order_points(paper_contour.reshape(4, 2))
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
    """Isolates the foot shape using edge-based contouring."""
    h, w = image.shape[:2]
    max_dim = 256.0
    scale = max_dim / max(h, w) if max(h, w) > max_dim else 1.0
    small_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    small_image = enhance_image(small_image)
    sh, sw = small_image.shape[:2]
    
    # Get paper boundaries in small image
    small_paper_rect = paper_rect * scale
    x, y, pw, ph = cv2.boundingRect(small_paper_rect.astype(np.int32))
    
    # Detect edges
    gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    dilated = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=2)
    
    # ROI for foot
    roi_w = int(sw * 0.65)
    if foot_side == "left":
        sx1, sx2 = max(0, x - roi_w), max(0, x - 5)
    else:
        sx1, sx2 = min(sw - 1, (x + pw) + 5), min(sw - 1, (x + pw) + roi_w)
        
    roi_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.rectangle(roi_mask, (sx1, 0), (sx2, sh), 255, -1)
    masked = cv2.bitwise_and(dilated, dilated, mask=roi_mask)
    
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise CVError("Foot not detected next to paper.")
        
    foot_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(foot_contour) < 500:
        raise CVError("Foot shape too small. Try re-positioning.")
        
    # Create filled mask and re-extract clean contour
    m = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(m, [foot_contour], -1, 255, -1)
    refined_c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    foot_contour = max(refined_c, key=cv2.contourArea)
    
    return (foot_contour / scale).astype(np.int32)

def measure_foot(foot_contour, M):
    """Calculates foot dimensions and applies 3D compensation."""
    c = np.array(foot_contour, dtype=np.float32)
    warped_c = cv2.perspectiveTransform(c, M)
    rect = cv2.minAreaRect(warped_c)
    (_, (w_px, h_px), _) = rect
    
    # 0.85 compensation is critical to prevent 'Extremely Large' results
    length_cm = (max(w_px, h_px) / PIXELS_PER_CM) * 0.85
    width_cm = (min(w_px, h_px) / PIXELS_PER_CM) * 0.85
    return length_cm, width_cm

def calculate_shoe_size(length_cm: float):
    """Ecommerce Standard Conversion (Paris Points)."""
    uk_raw = (length_cm - 18.0) / 0.846
    uk_size = max(1, int(round(uk_raw)))
    return uk_size, uk_size + 1

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
    if not (15 < l_cm < 35):
        raise CVError("Measurement failed. Ensure foot is correctly placed next to A4 paper.")
        
    uk, us = calculate_shoe_size(l_cm)
    return {"foot_side": foot_side, "length_cm": round(l_cm, 1), "width_cm": round(w_cm, 1), "shoe_size_uk": uk, "shoe_size_us": us}

def fast_validate_image(image: np.ndarray, foot_side: str) -> dict:
    """Lite version for real-time camera feedback."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 10.0:
            return {"valid": False, "message": "Too blurry"}
        rect, mw, mh = detect_a4_paper(image)
        return {"valid": True, "message": "Perfect! Hold still...", "confidence": 0.9}
    except:
        return {"valid": False, "message": "A4 Paper not detected"}
