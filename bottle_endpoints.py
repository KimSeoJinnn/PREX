"""Reobserve a cap and a terminating bottom rim inside a bottle proposal.

Only small endpoint neighborhoods are inspected. The hand-covered body is not
needed, and a detector box or a missing rim never becomes a measured endpoint.
"""

import cv2
import numpy as np

from bottle_landmarks import Landmarks


def cap_identity(frame, cap, width, blocked=None):
    x, y = np.rint(cap).astype(int)
    rx = max(4, round(width * .12))
    ry = max(3, round(width * .07))
    x0, x1 = max(0, x - rx), min(frame.shape[1], x + rx + 1)
    y0, y1 = max(0, y - ry), min(frame.shape[0], y + ry + 1)
    patch = frame[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    keep = np.ones(patch.shape[:2], bool) if blocked is None else blocked[y0:y1, x0:x1] == 0
    if np.count_nonzero(keep) < 20:
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[keep].astype(float)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[keep].astype(float)
    return np.r_[np.median(lab, axis=0), np.median(hsv[:, 1:], axis=0)]


def same_cap(identity, current):
    if identity is None or current is None:
        return False
    # Chroma tolerates exposure changes and a different background/hand pose.
    return (np.linalg.norm(identity[1:3] - current[1:3]) < 22
            and abs(identity[3] - current[3]) < 100
            and (identity[3] < 30 or current[3] >= max(15, identity[3] * .35)))


def foot_patch(frame, cap, base, width):
    """Compact material ABOVE the bottom; excludes hand/scene below the rim."""
    delta = base - cap
    length = float(np.linalg.norm(delta))
    direction = delta / length
    normal = np.array([direction[1], -direction[0]])
    along = np.linspace(-.080, -.015, 24) * length
    across = np.linspace(-.22, .22, 40) * width
    xy = base + along[:, None, None] * direction + across[None, :, None] * normal
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.remap(gray, xy[:, :, 0].astype(np.float32), xy[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)


def same_foot(reference, current):
    if abs(float(np.mean(reference)) - float(np.mean(current))) > 30:
        return False
    if np.std(reference) < 3 or np.std(current) < 3:
        return False
    score = float(cv2.matchTemplate(current, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
    return np.isfinite(score) and score >= .30


def _cap(frame, seed, width, length, blocked, identity):
    radius = np.array([max(width * .70, 24), max(length * .14, 24)])
    lo = np.maximum(0, np.floor(seed - radius)).astype(int)
    hi = np.minimum(frame.shape[1::-1], np.ceil(seed + radius + 1)).astype(int)
    patch = frame[lo[1]:hi[1], lo[0]:hi[0]]
    if patch.size == 0 or identity is None:
        return None
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(float)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    if identity[3] < 25:
        # Achromatic caps need actual boundary contrast; an arbitrary white
        # patch is not evidence of a cap. Use compact closed edge contours.
        gray = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        mask = cv2.Canny(gray, 20, 55)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    else:
        compatible = np.linalg.norm(lab[:, :, 1:3] - identity[1:3], axis=2) < 13
        compatible &= hsv[:, :, 1] >= max(15, identity[3] * .60)
        compatible &= hsv[:, :, 2] >= max(25, identity[4] * .65)
        mask = compatible.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask[blocked[lo[1]:hi[1], lo[0]:hi[0]] > 0] = 0
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        (cx, cy), (a, b), _ = cv2.minAreaRect(contour)
        major, minor = max(a, b), min(a, b)
        area = abs(cv2.contourArea(contour))
        if not (max(12, width * .20) <= major <= width * .90 and 5 <= minor <= length * .16):
            continue
        if not 1.05 <= major / max(minor, 1) <= 5.5 or area < max(35, major * minor * .45):
            continue
        center = np.array([cx, cy]) + lo
        if np.linalg.norm((center - seed) / radius) > .95:
            continue
        descriptor = cap_identity(frame, center, width, blocked)
        if not same_cap(identity, descriptor):
            continue
        compact = min(1., area / max(major * minor, 1))
        quality = .65 * compact + .35 * (1 - min(1., np.linalg.norm((center - seed) / radius)))
        found.append((quality, center))
    if not found:
        return None
    found.sort(key=lambda item: item[0], reverse=True)
    # Nested rim/color contours can describe the same cap. A second, spatially
    # separate equally good cap makes the observation ambiguous.
    if any(q > found[0][0] - .08 and np.linalg.norm(c - found[0][1]) > width * .35 for q, c in found[1:]):
        return None
    return found[0][1], float(found[0][0])


def _rim(frame, cap, base, width, blocked):
    delta = base - cap
    length = float(np.linalg.norm(delta))
    direction = delta / length
    rotation = np.stack(([direction[1], -direction[0]], direction)) * (320. / length)
    matrix = np.column_stack((rotation, np.array([160., 32.]) - rotation @ cap)).astype(np.float32)
    aligned = cv2.warpAffine(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), matrix, (320, 440))
    excluded = cv2.warpAffine(blocked, matrix, (320, 440), flags=cv2.INTER_NEAREST)
    inside = cv2.warpAffine(np.full(frame.shape[:2], 255, np.uint8), matrix, (320, 440), flags=cv2.INTER_NEAREST)
    excluded[inside == 0] = 255
    smooth = cv2.GaussianBlur(aligned, (3, 3), 0)
    gx = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 1, 0))
    gy = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    gy[gx > gy * 1.8] = 0
    gx[excluded > 0] = gy[excluded > 0] = 0
    w = float(np.clip(width * 320 / length, 35, 210))
    x0, x1 = round(160 - w * .43), round(160 + w * .43) + 1
    # Clear PET feet are often cropped short by the generic object detector.
    # Search farther down, but accept only a terminating rim in the image.
    rows = np.arange(324, 411)
    # Evaluate all rim rows together; no per-row Python image scans on the
    # hand-analysis worker.
    free = cv2.boxFilter((excluded == 0).astype(np.float32), -1, (1, 9))
    peaks = cv2.dilate(gy, np.ones((9, 1), np.uint8))[rows, x0:x1]
    support = peaks >= 12
    mid = support.shape[1] // 2
    coverage = np.mean(support, axis=1)
    valid = (np.mean(free[rows, x0:x1], axis=1) >= .85) & (coverage >= .45)
    valid &= (np.mean(support[:, :mid], axis=1) >= .32) & (np.mean(support[:, mid:], axis=1) >= .32)
    # A terminating rim loses body structure below it; internal rings don't.
    energy = np.mean(np.minimum((gx + gy)[:, x0:x1] / 25, 1), axis=1)
    cumulative = np.r_[0., np.cumsum(energy)]
    below = (cumulative[rows + 25] - cumulative[rows + 10]) / 15
    above = (cumulative[rows - 10] - cumulative[rows - 25]) / 15
    # An interior rib can have strong horizontal contrast and locally ending
    # highlights on both sides. It is not the bottom if body texture continues
    # below it across the central strip.
    valid &= (above >= .18) & (below < above * .78)
    # Require body sides immediately above the rim, not a long window that
    # still contains the old bottle while a lower background line wins.
    side_strength = cv2.boxFilter(np.minimum(gx / 25, 2), -1, (1, 7), anchor=(0, 6))
    below_strength = cv2.boxFilter(np.minimum(gx / 25, 2), -1, (1, 9), anchor=(0, 0))
    sides = []
    for sign in (-1, 1):
        center = 160 + sign * w * .48
        lo, hi = max(1, round(center - w * .35)), min(319, round(center + w * .35) + 1)
        strengths = side_strength[rows - 5, lo:hi]
        continuing = below_strength[rows + 10, lo:hi]
        ending = (strengths >= .35) & (continuing < np.maximum(.25, strengths * .5))
        # Window/frame/background lines often outshine a clear bottle side,
        # but continue BELOW its bottom. They cannot define the bottle width.
        indices = np.argmax(np.where(ending, strengths, 0), axis=1)
        valid &= ending[np.arange(len(rows)), indices]
        sides.append(indices + lo)
    span = sides[1] - sides[0]
    valid &= (span >= w * .55) & (span <= w * 1.4)
    quality = .60 * coverage + .25 * (1 - below) + .15 * np.exp(-.5 * ((rows - 352) / 22) ** 2)
    quality[~valid] = -1
    best = int(np.argmax(quality))
    if quality[best] < .62:
        return None
    cx = (sides[0][best] + sides[1][best]) / 2
    row = rows[best]
    strip = gy[row - 4:row + 5, x0:x1]
    maxima = np.argmax(strip, axis=0)
    strong = np.max(strip, axis=0) >= 12
    cy = float(np.median(maxima[strong] + row - 4))
    point = cv2.transform(np.array([[[cx, cy]]], np.float32), cv2.invertAffineTransform(matrix))[0, 0]
    # A generic bottle box may include the palm. The paired sides just above
    # the observed foot give a better search-width hint than that box does.
    # This is initialization geometry, never a deformation measurement.
    body_width = float(span[best] * length / 320.)
    return point.astype(float), float(quality[best]), body_width


def observe_endpoints(frame, seed, blocked, identity=None, region=None):
    """Return fresh image evidence for BOTH endpoints, or None; no extrapolation."""
    length = float(np.linalg.norm(seed.base - seed.cap))
    if length < 80 or not np.isfinite([seed.cap, seed.base]).all():
        return None
    identity = cap_identity(frame, seed.cap, seed.body_width, blocked) if identity is None else identity
    found = _cap(frame, seed.cap, seed.body_width, length, blocked, identity)
    if found is None:
        return None
    cap, cap_quality = found
    base_seed = seed.base + (cap - seed.cap)
    rim = _rim(frame, cap, base_seed, seed.body_width, blocked)
    if rim is None:
        return None
    base, base_quality, observed_width = rim
    if region is not None:
        x, y, w, h = region
        if not x - 6 <= base[0] <= x + w + 6 or not y <= base[1] <= y + h + max(6, h * .08):
            return None
    new_length = float(np.linalg.norm(base - cap))
    if not .78 <= new_length / length <= 1.22:
        return None
    for point in (cap, base):
        x, y = np.rint(point).astype(int)
        if x < 8 or y < 8 or x >= frame.shape[1] - 8 or y >= frame.shape[0] - 8:
            return None
        if np.mean(blocked[y - 5:y + 6, x - 5:x + 6] > 0) > .20:
            return None
    return Landmarks(cap, base, observed_width, min(seed.score, cap_quality, base_quality),
                     'learned_observed_endpoints', (cap_quality, base_quality))
