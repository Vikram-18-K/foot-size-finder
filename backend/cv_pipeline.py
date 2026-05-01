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

def validate_image_quality(image: np.ndarray, threshold: float = 15.0):
    """
    Rejects blurry images using Laplacian variance.
    Expects variance to be above the given threshold.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < threshold:
        raise CVError(f"Image is too blurry. Please hold the camera steady.")
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
    Detects the A4 reference paper in the image.
    Validates it is a 4-sided polygon with correct proportions and acceptable tilt.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise CVError("No objects detected. Ensure A4 paper is in the frame.")
        
    # Sort by area, descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    paper_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        # Loosen approximation slightly to ignore small jagged edges
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        
        # Look for a large 4-sided polygon
        if len(approx) == 4 and cv2.contourArea(approx) > 5000:
            paper_contour = approx
            break
            
    if paper_contour is None:
        raise CVError("A4 paper not cleanly detected. Make sure the entire paper is visible.")
        
    pts = paper_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Perspective Tilt Validation
    width_ratio = min(widthA, widthB) / max(widthA, widthB)
    height_ratio = min(heightA, heightB) / max(heightA, heightB)
    
    # Loosened perspective distortion tolerance
    if width_ratio < 0.65 or height_ratio < 0.65:
        raise CVError("Camera angle is too tilted. Please hold the phone more parallel to the floor.")
        
    # Aspect Ratio validation (A4 is ~1.414)
    aspect_ratio = maxHeight / float(maxWidth) if maxWidth > 0 else 0
    
    # Loosened aspect ratio validation
    if not (1.0 < aspect_ratio < 1.9) and not (0.5 < aspect_ratio < 1.0):
        raise CVError("Detected reference object does not match A4 paper proportions.")
        
    return rect, maxWidth, maxHeight

def get_homography(rect, maxWidth, maxHeight):
    """
    Computes the perspective transform matrix from the detected paper
    to a standardized coordinate system mapping pixels to real centimeters.
    """
    # Determine orientation (Landscape vs Portrait)
    if maxWidth > maxHeight:
        paper_w_cm, paper_h_cm = A4_LENGTH_CM, A4_WIDTH_CM
    else:
        paper_w_cm, paper_h_cm = A4_WIDTH_CM, A4_LENGTH_CM
        
    # Destination points in our new standardized metric space
    dst_w = paper_w_cm * PIXELS_PER_CM
    dst_h = paper_h_cm * PIXELS_PER_CM
    
    dst = np.array([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    return M

def segment_foot(image: np.ndarray, paper_rect: np.ndarray, foot_side: str):
    """
    Robust foot segmentation using GrabCut.
    Explicitly handles floor textures by initializing foreground next to the A4 paper.
    """
    h, w = image.shape[:2]
    
    # Fast processing: downsample if image is very large
    max_dim = 640.0
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        
    small_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    sh, sw = small_image.shape[:2]
    
    # Scale paper rect
    small_paper_rect = paper_rect * scale
    
    # Initialize GrabCut mask
    mask = np.zeros((sh, sw), np.uint8)
    mask[:] = cv2.GC_PR_BGD  # Probable background everywhere
    
    # A4 Paper is Definite Background
    cv2.fillPoly(mask, [small_paper_rect.astype(np.int32)], cv2.GC_BGD)
    
    # Get paper center to define probable foreground
    M = cv2.moments(small_paper_rect.astype(np.int32))
    if M["m00"] == 0:
        raise CVError("Failed to calculate paper dimensions.")
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    paper_width = np.linalg.norm(small_paper_rect[0] - small_paper_rect[1])
    paper_height = np.linalg.norm(small_paper_rect[1] - small_paper_rect[2])
    
    # Define bounding box for the foot
    foot_w = int(paper_width * 1.5)
    foot_h = int(paper_height * 1.2)
    
    if foot_side == "left":
        # Left foot means foot is on the left side of the paper
        x1 = max(0, cx - int(paper_width*0.3) - foot_w)
        x2 = max(0, cx - int(paper_width*0.3))
    else:
        # Right foot means foot is on the right side of the paper
        x1 = min(sw, cx + int(paper_width*0.3))
        x2 = min(sw, cx + int(paper_width*0.3) + foot_w)
        
    y1 = max(0, cy - int(foot_h/2))
    y2 = min(sh, cy + int(foot_h/2))
    
    # Mark the foot area as probable foreground
    cv2.rectangle(mask, (x1, y1), (x2, y2), cv2.GC_PR_FGD, -1)
    
    # Edges of the image are definite background to prevent floor from taking over
    margin = int(max_dim * 0.05)
    cv2.rectangle(mask, (0, 0), (sw, sh), cv2.GC_BGD, margin)
    
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    try:
        cv2.grabCut(small_image, mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        raise CVError("Could not isolate foot. Ensure foot is clearly separated from background.")
        
    # Extract foreground
    mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    
    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Upscale mask back to original resolution
    full_mask = cv2.resize(mask2, (w, h), interpolation=cv2.INTER_NEAREST)
    
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise CVError("Foot not detected next to the paper.")
        
    # Filter valid contours
    valid_contours = [c for c in contours if cv2.contourArea(c) > (5000 / (scale * scale))]
    
    if not valid_contours:
        raise CVError("Detected object too small to be a foot.")
        
    foot_contour = max(valid_contours, key=cv2.contourArea)
    return foot_contour

def measure_foot(foot_contour, M):
    """
    Applies the homography matrix to warp the foot contour into the flat, standardized space.
    Calculates accurate length/width utilizing rotation-invariant minimum area rectangles.
    """
    # Apply homography purely to the contour coordinates
    foot_contour_float = np.array(foot_contour, dtype=np.float32)
    transformed_contour = cv2.perspectiveTransform(foot_contour_float, M)
    transformed_contour = np.array(transformed_contour, dtype=np.float32)
    
    # Get rotation-invariant bounding rectangle
    rect = cv2.minAreaRect(transformed_contour)
    (center, (width_px, height_px), angle) = rect
    
    # The foot length is naturally the longer dimension
    length_px = max(width_px, height_px)
    width_px = min(width_px, height_px)
    
    # Scale from warped pixels to centimeters
    length_cm = length_px / PIXELS_PER_CM
    width_cm = width_px / PIXELS_PER_CM
    
    # 3D Projection Compensation (2% reduction to account for ankle/heel height projecting outward)
    length_cm = length_cm * 0.98
    
    return length_cm, width_cm

def calculate_shoe_size(length_cm: float):
    """
    Converts metric foot length to standard shoe sizes based on the Brannock device formula.
    Formula: Size = 3 * (Foot Length in inches) - 22 (for US Men's)
    """
    length_inches = length_cm / 2.54
    
    us_raw = (3 * length_inches) - 22
    uk_raw = us_raw - 1
    
    # Round to nearest 0.5
    us_size = max(1.0, round(us_raw * 2.0) / 2.0)
    uk_size = max(1.0, round(uk_raw * 2.0) / 2.0)
    
    return uk_size, us_size

def process_image(image: np.ndarray, foot_side: str) -> dict:
    """
    Master pipeline orchestrator.
    Executes geometric validation, detection, homography, segmentation, and measurement.
    """
    # 1. Reject blurred images
    validate_image_quality(image)
    
    # 2. Detect reference and validate camera tilt
    rect, maxWidth, maxHeight = detect_a4_paper(image)
    
    # 3. Calculate scaling transform
    M = get_homography(rect, maxWidth, maxHeight)
    
    # 4. Isolate foot using robust GrabCut
    foot_contour = segment_foot(image, rect, foot_side)
    
    # 5. Extract accurate metrics via rotation-invariant transform
    length_cm, width_cm = measure_foot(foot_contour, M)
    
    # Post-validation sanity check
    if length_cm < 10 or length_cm > 35:
        raise CVError(f"Calculated length ({length_cm:.1f} cm) is physically unlikely. Please adjust lighting and re-scan.")
        
    uk_size, us_size = calculate_shoe_size(length_cm)
    
    return {
        "length_cm": round(length_cm, 1),
        "width_cm": round(width_cm, 1),
        "shoe_size_uk": uk_size,
        "shoe_size_us": us_size
    }

def fast_validate_image(image: np.ndarray, foot_side: str) -> dict:
    """
    Lightweight, sub-100ms validation endpoint.
    Priority flow: Blur -> A4 -> Tilt -> Foot
    """
    response = {
        "a4_detected": False,
        "foot_detected": False,
        "is_blurry": False,
        "tilt_ok": False,
        "valid": False,
        "confidence": 0.0,
        "message": ""
    }
    
    # Downsample image for lightning-fast validation processing
    h, w = image.shape[:2]
    max_dim = 640
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Blur Detection (Laplacian Variance)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 15.0:  # Loosened from 25.0
        response["is_blurry"] = True
        response["message"] = "Image is too blurry. Please hold steady."
        return response

    # 2. A4 Detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    edged = cv2.dilate(edged, None, iterations=1)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        response["message"] = "A4 paper not detected."
        return response
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    paper_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)  # Loosened epsilon
        if len(approx) == 4 and cv2.contourArea(approx) > 1000: # Loosened scaled threshold
            paper_contour = approx
            break
            
    if paper_contour is None:
        response["message"] = "A4 paper not cleanly detected."
        return response
        
    response["a4_detected"] = True

    # 3. Tilt Validation
    pts = paper_contour.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    width_ratio = min(widthA, widthB) / max(widthA, widthB)
    height_ratio = min(heightA, heightB) / max(heightA, heightB)
    
    if width_ratio < 0.55 or height_ratio < 0.55:  # Loosened from 0.70
        response["message"] = "Camera angle is too tilted."
        return response
        
    response["tilt_ok"] = True

    # 4. Robust Foot Detection (Edges & Positioning, disregarding skin color)
    mask = np.ones(gray.shape, dtype=np.uint8) * 255
    cv2.fillPoly(mask, [rect.astype(np.int32)], 0) # Mask out A4
    
    # Adaptive thresholding isolates objects strictly outside the paper bounds
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    # Morphological closing to group foot features
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    foot_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    max_area = 0
    for c in foot_contours:
        area = cv2.contourArea(c)
        if area > 1500:  # Loosened scaled footprint threshold from 4000
            response["foot_detected"] = True
            if area > max_area:
                max_area = area
                
    if not response["foot_detected"]:
        response["message"] = "Foot not detected next to paper."
        return response
        
    # Validation Passed
    # Confidence Score: Harmonic combination of edge sharpness and foot coverage ratio
    conf_variance = min(1.0, variance / 150.0)
    conf_area = min(1.0, max_area / (maxWidth * maxHeight * 2.0))
    confidence = (conf_variance * 0.6) + (conf_area * 0.4)

    response["confidence"] = round(max(0.1, min(confidence, 0.99)), 2)
    response["valid"] = True
    response["message"] = "Perfect! Hold still..."
    
    return response
