"""Automatic cap/rim landmarks from image evidence, with no added models.

This is a constrained bottle-shape detector, not a trained semantic keypoint
network. A cap must be followed by a wider body and a visible terminating rim.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from bottle_outline import hand_mask


@dataclass(frozen=True)
class Landmarks:
    cap: np.ndarray
    base: np.ndarray
    body_width: float
    score: float
    source: str
    endpoint_quality: tuple[float, float] | None = None


def _cap_candidates(frame, blocked, region=None):
    height, width = frame.shape[:2]
    x, y, w, h = region if region is not None else (0, 0, width, height)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, int(x + w)), min(height, int(y + h))
    if x1 <= x0 or y1 <= y0:
        return []
    patch = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2., tileGridSize=(6, 6)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(enhanced, (3, 3), 0), 15, 50)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    color = ((hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    masks = (color, edges)
    candidates = []
    for kind, mask in enumerate(masks):
        mask[blocked[y0:y1, x0:x1] > 0] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            cw, ch = sorted(rect[1], reverse=True)
            if not 15 <= cw <= min(125, width * .32) or not 6 <= ch <= 65:
                continue
            aspect = cw / ch
            if not 1.25 <= aspect <= 5.5 or abs(cv2.contourArea(contour)) < cw * ch * .40:
                continue
            center = np.array(rect[0], float) + (x0, y0)
            if region is not None and center[1] > y + h * .36:
                continue
            box = cv2.boxPoints(rect)
            vectors = np.roll(box, -1, axis=0) - box
            major = vectors[np.argmax(np.linalg.norm(vectors, axis=1))]
            major /= np.linalg.norm(major)
            direction = np.array([-major[1], major[0]], float)
            if direction[1] < 0:
                direction = -direction
            if direction[1] < .65:
                continue
            cx, cy = np.rint(center).astype(int)
            if cx < 8 or cy < 8 or cx >= width - 8 or cy >= height - 8:
                continue
            priority = (1.2 if kind == 0 else 1.) / (1 + abs(aspect - 2.4) * .2)
            candidates.append((priority, center, direction, float(cw)))
    # A white cap's contour may connect to the neck/body and cease to be a
    # separate component. Paired rim lines still provide a color-free seed.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 12, minLineLength=15, maxLineGap=5)
    segments = []
    if lines is not None:
        for line in lines[:120, 0]:
            first, last = line[:2].astype(float), line[2:].astype(float)
            vector = last - first
            length = float(np.linalg.norm(vector))
            if not 15 <= length <= min(125, width * .32) or abs(vector[0]) < abs(vector[1]) * 1.3:
                continue
            if vector[0] < 0:
                vector = -vector
            segments.append(((first + last) / 2, vector / length, length))
    for index, (center_a, major_a, width_a) in enumerate(segments):
        for center_b, major_b, width_b in segments[index + 1:]:
            if float(major_a @ major_b) < .99 or min(width_a, width_b) < max(width_a, width_b) * .65:
                continue
            major = major_a + major_b
            major /= np.linalg.norm(major)
            direction = np.array([-major[1], major[0]])
            delta = center_b - center_a
            cw = (width_a + width_b) / 2
            ch = abs(float(delta @ direction))
            if not 6 <= ch <= 65 or not 1.25 <= cw / ch <= 5.5 or abs(float(delta @ major)) > cw * .25:
                continue
            center = (center_a + center_b) / 2 + (x0, y0)
            if region is not None and center[1] > y + h * .36:
                continue
            cx, cy = np.rint(center).astype(int)
            if cx < 8 or cy < 8 or cx >= width - 8 or cy >= height - 8 or blocked[cy, cx]:
                continue
            priority = .9 / (1 + abs(cw / ch - 2.4) * .2)
            candidates.append((priority, center, direction, cw))
    candidates.sort(key=lambda item: item[0], reverse=True)
    unique = []
    for item in candidates:
        if any(np.linalg.norm(item[1] - other[1]) < .35 * min(item[3], other[3]) for other in unique):
            continue
        unique.append(item)
        if len(unique) == 10:
            break
    return unique


def _body_and_rim(frame, blocked, cap, direction, cap_width, source):
    normal = np.array([direction[1], -direction[0]])
    half = int(np.ceil(cap_width * 2.1))
    top = int(np.ceil(cap_width * .5))
    height = min(680, int(np.ceil(cap_width * 10.5)))
    rotation = np.stack((normal, direction))
    matrix = np.column_stack((rotation, np.array([half, top]) - rotation @ cap)).astype(np.float32)
    size = (2 * half + 1, height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.warpAffine(gray, matrix, size)
    gray = cv2.createCLAHE(clipLimit=2., tileGridSize=(6, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    mask = cv2.warpAffine(blocked, matrix, size, flags=cv2.INTER_NEAREST)
    inside = cv2.warpAffine(np.full(frame.shape[:2], 255, np.uint8), matrix, size,
                            flags=cv2.INTER_NEAREST)
    mask[inside == 0] = 255
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    vertical_edges = gx.copy()
    vertical_edges[(gy > gx * 1.8) | (mask > 0)] = 0
    yy = np.arange(height)
    search_left = (max(1, round(half - cap_width * 1.8)), round(half - cap_width * .5))
    search_right = (round(half + cap_width * .5), min(size[0] - 1, round(half + cap_width * 1.8)))
    locations = []
    strengths = []
    for lo, hi in (search_left, search_right):
        window = vertical_edges[:, lo:hi]
        if not window.shape[1]:
            return None
        # Saturation keeps a single bright reflection from dominating a rim.
        columns = np.arange(lo, hi)
        expected = half + (-1 if lo < half else 1) * cap_width * .95
        prior = np.exp(-.5 * ((columns - expected) / (cap_width * .65)) ** 2)
        peaks = np.argmax(np.minimum(window / 30, 3) * prior, axis=1)
        locations.append(peaks + lo)
        strengths.append(window[yy, peaks])
    left, right = locations
    widths = right - left
    centers = (left + right) / 2
    paired = (strengths[0] >= 12) & (strengths[1] >= 12)
    paired &= (widths >= cap_width * 1.15) & (widths <= cap_width * 3.6)
    paired &= np.abs(centers - half) <= cap_width * .6
    paired[:round(top + cap_width * 1.1)] = False
    usable = np.flatnonzero(paired)
    if len(usable) < max(12, cap_width * .8):
        return None
    body_width = float(np.median(widths[paired]))
    # A real terminating rim has support above it and loses the body below it.
    # Internal label lines and transient gaps therefore do not qualify alone.
    best = None
    for row in range(round(top + cap_width * 3.1), min(height - 12, round(top + cap_width * 10))):
        before = paired[max(top, row - round(cap_width * 1.2)):row - 3]
        after = paired[row + round(cap_width * .25):min(height, row + round(cap_width * 1.1))]
        farther = paired[row + round(cap_width * .5):min(height, row + round(cap_width * 2.0))]
        if len(before) < 10 or len(after) < 10 or np.mean(before) < .4:
            continue
        if np.mean(after) > .20 or (len(farther) and np.mean(farther) > .22):
            continue
        body_rows = paired[round(top + cap_width):row]
        if len(body_rows) < cap_width * 2 or np.mean(body_rows) < .35:
            continue
        lo, hi = round(half - body_width * .6), round(half + body_width * .6)
        y0, y1 = max(0, row - round(cap_width * .20)), min(height, row + round(cap_width * .20) + 1)
        rim = gy[y0:y1, max(0, lo):min(size[0], hi)].copy()
        rim[mask[y0:y1, max(0, lo):min(size[0], hi)] > 0] = 0
        row_support = np.mean(rim >= 15, axis=1)
        if not len(row_support) or float(np.max(row_support)) < .28:
            continue
        supported_rows = np.flatnonzero(row_support >= max(.20, float(np.max(row_support)) * .55))
        base_y = y0 + float(np.average(supported_rows, weights=row_support[supported_rows]))
        near = paired[max(top, row - round(cap_width * .7)):row]
        near_centers = centers[max(top, row - round(cap_width * .7)):row][near]
        base_x = float(np.median(near_centers)) if len(near_centers) else half
        score = float(.45 * np.mean(body_rows) + .35 * min(1, float(np.max(row_support)) / .6)
                      + .20 * (1 - np.mean(after)))
        if best is None or score > best[0]:
            best = (score, base_x, base_y)
    if best is None:
        return None
    score, bx, by = best
    inverse = cv2.invertAffineTransform(matrix)
    base = cv2.transform(np.array([[[bx, by]]], np.float32), inverse)[0, 0].astype(float)
    if np.linalg.norm(base - cap) < 90 or score < .62:
        return None
    if np.any(base < 8) or base[0] >= frame.shape[1] - 8 or base[1] >= frame.shape[0] - 8:
        return None
    return Landmarks(cap, base, body_width, score, source)


def detect_landmarks(frame, hand, region=None, expected=None, blocked=None):
    blocked = hand_mask(frame.shape, hand) if blocked is None else blocked
    results = []
    for _, cap, direction, cap_width in _cap_candidates(frame, blocked, region):
        result = _body_and_rim(frame, blocked, cap, direction, cap_width,
                               "automatic_shape" if region is None else "automatic_detector_shape")
        if result is None:
            continue
        if expected is not None:
            old_cap, old_base = expected
            length = np.linalg.norm(old_base - old_cap)
            if max(np.linalg.norm(result.cap - old_cap), np.linalg.norm(result.base - old_base)) > length * .28:
                continue
        if any(np.linalg.norm(result.cap - other.cap) < 15 and np.linalg.norm(result.base - other.base) < 20 for other in results):
            continue
        results.append(result)
    results.sort(key=lambda item: item.score, reverse=True)
    return results
