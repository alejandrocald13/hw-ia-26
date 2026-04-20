import cv2
import numpy as np


def resize_to_fit(image, max_w, max_h):
    if image is None:
        return None

    h, w = image.shape[:2]
    if w == 0 or h == 0:
        return image

    scale = min(max_w / w, max_h / h)
    if scale <= 0:
        scale = 1

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def rotate_image(image, angle):
    if image is None:
        return None

    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)


def apply_rgb_balance(image, r_scale=255, g_scale=255, b_scale=255, mask=None):
    """
    r_scale, g_scale, b_scale en rango 0-255
    255 = canal normal
    0 = canal apagado
    """
    if image is None:
        return None

    result = image.copy().astype(np.float32)

    # OpenCV trabaja en BGR
    b_factor = b_scale / 255.0
    g_factor = g_scale / 255.0
    r_factor = r_scale / 255.0

    modified = result.copy()
    modified[:, :, 0] *= b_factor
    modified[:, :, 1] *= g_factor
    modified[:, :, 2] *= r_factor
    modified = np.clip(modified, 0, 255).astype(np.uint8)

    if mask is None:
        return modified

    output = image.copy()
    output[mask > 0] = modified[mask > 0]
    return output


def apply_gaussian_blur(image, intensity=0, mask=None):
    if image is None:
        return None

    if intensity <= 0:
        return image.copy()

    ksize = intensity * 2 + 1
    blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

    if mask is None:
        return blurred

    output = image.copy()
    output[mask > 0] = blurred[mask > 0]
    return output


def apply_sobel_x(image, intensity=0, mask=None):
    """
    intensity: 0-10
    0 = no aplicar
    """
    if image is None:
        return None

    if intensity <= 0:
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel = np.absolute(sobel)
    sobel = np.clip(sobel * intensity, 0, 255).astype(np.uint8)
    sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    if mask is None:
        return sobel_bgr

    output = image.copy()
    output[mask > 0] = sobel_bgr[mask > 0]
    return output


def apply_sobel_y(image, intensity=0, mask=None):
    """
    intensity: 0-10
    0 = no aplicar
    """
    if image is None:
        return None

    if intensity <= 0:
        return image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.absolute(sobel)
    sobel = np.clip(sobel * intensity, 0, 255).astype(np.uint8)
    sobel_bgr = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)

    if mask is None:
        return sobel_bgr

    output = image.copy()
    output[mask > 0] = sobel_bgr[mask > 0]
    return output


def clamp_point(x, y, width, height):
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    return x, y


def normalize_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)
    return left, top, right, bottom


def create_selection_mask(shape, image_shape, p1, p2):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1 = clamp_point(p1[0], p1[1], w, h)
    x2, y2 = clamp_point(p2[0], p2[1], w, h)

    if shape == "rect":
        left, top, right, bottom = normalize_points((x1, y1), (x2, y2))
        cv2.rectangle(mask, (left, top), (right, bottom), 255, -1)

    elif shape == "circle":
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        radius = int(max(abs(x2 - x1), abs(y2 - y1)) / 2)
        radius = max(1, radius)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

    return mask


def paint_region(image, rgb_color, mask, alpha=0.45):
    """
    Pinta la región con transparencia.
    rgb_color = (r, g, b)
    alpha: 0.0 a 1.0
    """
    if image is None or mask is None:
        return image.copy() if image is not None else None

    r, g, b = rgb_color
    overlay_color = np.array([b, g, r], dtype=np.float32)

    output = image.copy().astype(np.float32)
    region = mask > 0

    output[region] = (1 - alpha) * output[region] + alpha * overlay_color
    output = np.clip(output, 0, 255).astype(np.uint8)

    return output