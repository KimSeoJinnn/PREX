"""Conservative hand occlusion from landmarks and local image evidence.

This is an exclusion mask, not a bottle segmentation or an inferred silhouette.
It uses the observed hand's own chroma, rather than a fixed skin-color range.
"""

import cv2
import numpy as np

from bottle_outline import hand_mask


def body_occlusion_mask(frame, hand, geometry=None):
    geometry = hand_mask(frame.shape, hand) if geometry is None else geometry
    if len(hand) != 21:
        return geometry
    points = np.asarray(hand, float)
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    # During a wrap grip the detected bones often cover the fingers but miss
    # the fleshy palm/thumb pad beside the bottle. Grow connected hand color
    # farther sideways, without also swallowing the visible shoulder/foot.
    reach = int(np.clip(round(palm_width), 12, 80))
    vertical_reach = int(np.clip(round(palm_width * .28), 6, 24))
    yy, xx = np.nonzero(geometry)
    if not len(xx):
        return geometry
    x0, x1 = max(0, int(xx.min()) - reach), min(frame.shape[1], int(xx.max()) + reach + 1)
    y0, y1 = max(0, int(yy.min()) - vertical_reach), min(frame.shape[0], int(yy.max()) + vertical_reach + 1)
    local = geometry[y0:y1, x0:x1]
    patch = frame[y0:y1, x0:x1]
    chroma = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[:, :, 1:]
    seeds = cv2.erode(local, np.ones((5, 5), np.uint8)) > 0
    color = chroma[seeds].astype(float)
    # Neutral glass/background pixels inside the landmark polygon must not
    # teach the mask to flood all gray surfaces. Low-chroma images fall back
    # to the geometric mask; no hand boundary is invented from color alone.
    color = color[np.linalg.norm(color - 128, axis=1) >= 8]
    if len(color) < 20:
        return cv2.dilate(geometry, np.ones((5, 5), np.uint8))
    histogram, _, _ = np.histogram2d(color[:, 0], color[:, 1], bins=64, range=((0, 256), (0, 256)))
    histogram = cv2.GaussianBlur(histogram.astype(np.float32), (5, 5), 1.)
    indices = chroma.astype(np.int32) // 4
    compatible = histogram[indices[:, :, 0], indices[:, :, 1]] >= histogram.max() * .08
    # Distance transform avoids a large round-kernel dilation each frame.
    neighborhood = cv2.distanceTransform((local == 0).astype(np.uint8), cv2.DIST_L2, 3) <= reach
    compatible &= neighborhood
    compatible = cv2.morphologyEx(compatible.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels = cv2.connectedComponents(compatible, connectivity=8)
    connected = np.zeros(count, bool)
    connected[np.unique(labels[seeds & (compatible > 0)])] = True
    connected[0] = False
    grown = np.where(connected[labels] | (local > 0), 255, 0).astype(np.uint8)
    result = geometry.copy()
    result[y0:y1, x0:x1] = grown
    # Sobel/Gaussian support spills a few pixels beyond the skin boundary.
    return cv2.dilate(result, np.ones((5, 5), np.uint8))
