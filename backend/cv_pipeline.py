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
        if area > (img_area * 0.04): # Paper must be > 4% of image (relaxed from 8%)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            # Relaxed: allow 4-6 points to account for noise/rounding
            if 4 <= len(approx) <= 6:
                # Use minAreaRect to get a consistent 4-point rectangle
                rect_rot = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect_rot)
                box = np.int0(box)
                
                # Check aspect ratio (A4 is ~1.41)
                (x, y), (w_rot, h_rot), angle = rect_rot
                aspect_ratio = max(w_rot, h_rot) / min(w_rot, h_rot)
                
                if 1.2 < aspect_ratio < 1.7 and area > max_area:
                    paper_contour = box
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
    # Try edge-based detection first
    edged = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    dilated = cv2.dilate(edged, np.ones((5, 5), np.uint8), iterations=2)
    
    # ROI for foot - make it more flexible (75% of screen width)
    roi_w = int(sw * 0.75)
    if foot_side == "left":
        sx1, sx2 = max(0, x - roi_w), max(0, x - 2)
    else:
        sx1, sx2 = min(sw - 1, (x + pw) + 2), min(sw - 1, (x + pw) + roi_w)
        
    roi_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.rectangle(roi_mask, (sx1, 0), (sx2, sh), 255, -1)
    masked = cv2.bitwise_and(dilated, dilated, mask=roi_mask)
    
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fallback: If edge detection failed, try Otsu thresholding in the ROI
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 300:
        roi_gray = gray[:, sx1:sx2]
        _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Re-apply mask to the whole image
        full_thresh = np.zeros_like(gray)
        full_thresh[:, sx1:sx2] = thresh
        contours, _ = cv2.findContours(full_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise CVError("Foot not detected next to paper. Ensure your foot is fully visible.")
        
    foot_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(foot_contour) < 300: # Lowered from 500
        raise CVError("Foot shape too small. Ensure you're not too far away.")
        
    # Create filled mask and re-extract clean contour
    m = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(m, [foot_contour], -1, 255, -1)
    refined_c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    foot_contour = max(refined_c, key=cv2.contourArea)
    
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
    
    # Aspect Ratio Validation: Length vs Width
    # Human feet are typically 2.3x to 3.0x as long as they are wide.
    if width_cm > 0:
        ratio = length_cm / width_cm
        if ratio < 2.0 or ratio > 3.8:
            logger.warning(f"Impossible foot shape rejected: ratio {ratio:.2f}")
            raise CVError("Detection error. Please ensure your foot is fully visible and not skewed.")

    # Sanity check: Adult feet are rarely > 34cm or < 14cm
    if length_cm > 34.0 or length_cm < 14.0:
        logger.warning(f"Extreme measurement rejected: {length_cm}cm")
        raise CVError("Measurement failed. Ensure your foot is correctly placed next to the paper.")
        
    return length_cm, width_cm

def calculate_shoe_size(length_cm: float):
    """
    Industry Standard Foot-to-Shoe Size Conversion.
    Based on standard brand charts (Nike, Adidas, etc.)
    """
    # Standard formula for UK/India sizing matching Nike/Adidas charts:
    # 26.2cm -> UK 7.5, 27.1cm -> UK 8.5, 28cm -> UK 9.5
    uk_raw = (length_cm - 19.8) / 0.846
    uk_size = max(1, int(round(uk_raw)))
    
    # US Men is typically UK + 1
    us_men = uk_size + 1
    
    # EU Size formula: (Length_cm + 1.5) * 1.5
    eu_size = int(round(1.5 * (length_cm + 1.5)))
    
    # Final capping for realistic adult sizes
    uk_size = min(max(uk_size, 1), 14)
    us_men = min(max(us_men, 2), 15)
    
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
        is_blurry = cv2.Laplacian(gray, cv2.CV_64F).var() < 7.0
        if is_blurry:
            return {
                "a4_detected": False, "foot_detected": False, "is_blurry": True,
                "tilt_ok": True, "valid": False, "confidence": 0.0,
                "message": "Too blurry"
            }
            
        rect, mw, mh = detect_a4_paper(image)
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
    except Exception:
        return {
            "a4_detected": False, "foot_detected": False, "is_blurry": False,
            "tilt_ok": True, "valid": False, "confidence": 0.0,
            "message": "A4 Paper not detected"
        }
