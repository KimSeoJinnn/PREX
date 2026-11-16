"""Explicitly calibrated hand-guided proxy contours, never measured bottle edges.

Visible bottle edges above/below the grip anchor a linear rest contour. Fixed
landmark pairs on the rest hand hull carry its lateral offsets during a squeeze.
The hidden region remains an estimate; no sensor force or volume is recovered.
"""

import math

import cv2
import numpy as np


ESTIMATE_COLOR = (205, 106, 255)
JOINT_METRICS = (
    'joint_estimated_width_px', 'joint_estimated_width_reduction_pct',
    'joint_estimated_profiles',
)


def _canonical(pose, points):
    return cv2.transform(np.asarray(points, np.float32).reshape(-1, 1, 2), pose.matrix).reshape(-1, 2)


def _hull_links(hand, rows):
    """Freeze anatomical landmark pairs, rather than reselect fingers each frame."""
    hull = cv2.convexHull(hand.astype(np.float32), returnPoints=False).ravel()
    links = np.full((2, len(rows), 2), -1, int)
    weights = np.zeros((2, len(rows)))
    for row, y in enumerate(rows):
        hits = []
        for a, b in zip(hull, np.roll(hull, -1)):
            dy = hand[b, 1] - hand[a, 1]
            if abs(dy) < 1e-5:
                continue
            u = (y - hand[a, 1]) / dy
            if 0 <= u <= 1:
                hits.append((hand[a, 0] * (1 - u) + hand[b, 0] * u, a, b, u))
        if len(hits) >= 2:
            hits.sort()
            for side, hit in enumerate((hits[0], hits[-1])):
                links[side, row] = hit[1:3]
                weights[side, row] = hit[3]
    return links, weights


class JointContourEstimator:
    def __init__(self, station_rows, pixel_scale, body_width=120.):
        self.station_rows = np.asarray(station_rows, float)
        self.pixel_scale = pixel_scale
        self.body_width = float(body_width)
        self.reset()

    def reset(self):
        self.baseline = None
        self.starts_at = None
        self.samples = []
        self.stable_samples = 0
        self.observation_reason = 'show hand and bottle edges'
        self.status = 'H: relax body grip to set estimate'
        self.clear_current()

    @property
    def ready(self):
        return self.baseline is not None

    @property
    def calibrating(self):
        return self.starts_at is not None

    def clear_current(self):
        self.metrics = {name: None for name in JOINT_METRICS}
        self.profile_left = np.full(len(self.station_rows), np.nan)
        self.profile_right = self.profile_left.copy()
        self.segments = []
        self.anchor_source = 'none'
        self.display_only = False

    def request(self, now):
        self.reset()
        self.starts_at = now + 3.
        self.status = 'Relax BODY grip; estimate baseline in 3s'

    def missing(self, now, reason):
        self.clear_current()
        self.observation_reason = reason.removeprefix('Estimate paused: ')
        if self.calibrating:
            if now is None or not self.samples or now - self.samples[-1][0] > .5:
                self.samples.clear()
            self.stable_samples = 0
            self.status = 'Waiting: ' + self.observation_reason
        elif self.ready:
            self.status = reason

    def _observation(self, pose, hand, segments, blocked, require_anchors=True):
        self.observation_reason = 'show full hand'
        if len(hand) != 21:
            return None
        if not segments and require_anchors:
            self.observation_reason = 'show upper/lower bottle edges'
            return None
        hand = _canonical(pose, hand).astype(float)
        if not np.isfinite(hand).all() or len(np.unique(hand, axis=0)) < 8:
            return None
        palm = np.linalg.norm(hand[5] - hand[17])
        if not 15 <= palm <= 220:
            return None
        mask = pose.rectify(blocked, nearest=True)
        # The forearm can run below the bottle, outside its sides. Its bottom
        # must not become the bottom of the grip and veto all lower anchors.
        half = float(np.clip(self.body_width * .60, 25, 125))
        x0, x1 = max(25, round(160 - half)), min(295, round(160 + half) + 1)
        yy, xx = np.nonzero(mask[75:350, x0:x1])
        if len(yy) < 30:
            self.observation_reason = 'hold bottle BODY lightly'
            return None
        observed = (_canonical(pose, np.vstack(segments)).astype(float)
                    if segments else np.empty((0, 2)))
        if not require_anchors:
            # The proxy follows calibrated joint offsets in bottle coordinates.
            # Requiring all four edges on every frame defeats its purpose when
            # the grip hides those edges. The current axis still must be valid.
            self.observation_reason = 'tracking same grip with reference outline'
            return {'hand': hand, 'anchors': self.baseline['anchors'],
                    'mask': mask, 'observed': observed}
        top, bottom = float(yy.min() + 75), float(yy.max() + 75)
        if not 30 <= bottom - top <= 245:
            self.observation_reason = 'show bottle above/below hand'
            return None
        anchors = []
        for name, side in (('LEFT', observed[observed[:, 0] < 160]),
                           ('RIGHT', observed[observed[:, 0] >= 160])):
            side = side[np.abs(side[:, 0] - 160) <= self.body_width * .85]
            upper = side[(side[:, 1] < top - 3) & (side[:, 1] > max(70, top - 75))]
            lower = side[(side[:, 1] > bottom + 3) & (side[:, 1] < min(352, bottom + 75))]
            if len(upper) < 3:
                self.observation_reason = f'show {name} bottle edge ABOVE hand'
                return None
            if len(lower) < 3:
                self.observation_reason = f'show {name} bottle edge BELOW hand'
                return None
            upper = upper[np.argsort(upper[:, 1])[-6:]]
            lower = lower[np.argsort(lower[:, 1])[:6]]
            anchors.append([np.median(upper, axis=0), np.median(lower, axis=0)])
        anchors = np.asarray(anchors)
        if np.min(anchors[1, :, 0] - anchors[0, :, 0]) < 25:
            self.observation_reason = 'separate bottle from background'
            return None
        self.observation_reason = 'upper/lower edges found'
        return {'hand': hand, 'anchors': anchors, 'mask': mask, 'observed': observed}

    def _finish(self, now):
        recent = bool(self.samples) and now - self.samples[-1][0] < .5
        span = self.samples[-1][0] - self.samples[0][0] if self.samples else 0.
        if len(self.samples) < 4 or span < .5 or not recent:
            self.status = ('Waiting: hold relaxed BODY grip still' if recent
                           else 'Waiting: ' + self.observation_reason)
            self.samples.clear()
            return
        hand = np.median([s[1]['hand'] for s in self.samples], axis=0)
        anchors = np.median([s[1]['anchors'] for s in self.samples], axis=0)
        rows = np.linspace(max(anchors[:, 0, 1]), min(anchors[:, 1, 1]), 81)
        links, weights = _hull_links(hand, rows)
        eligible = np.all(links[:, :, 0] >= 0, axis=0)
        eligible[[0, -1]] = False
        if np.count_nonzero(eligible) < 10:
            self.status = 'Waiting: show full hand on bottle body'
            self.samples.clear()
            return
        reference = np.array([np.interp(rows, a[:, 1], a[:, 0]) for a in anchors])
        self.starts_at = None
        self.baseline = dict(hand=hand, anchors=anchors, rows=rows, links=links,
                             weights=weights, eligible=eligible, reference=reference,
                             hand_side=self.samples[-1][1].get('hand_side'))
        angle_samples = [s[1]['angles'] for s in self.samples if s[1].get('angles') is not None]
        self.baseline['angles'] = np.median(angle_samples, axis=0) if len(angle_samples) >= 4 else None
        self.samples.clear()
        self.status = 'Joint estimate ready; squeeze gently first'

    def update(self, pose, hand, segments, blocked, now, hand_side=None, display_only=False, angles=None):
        self.clear_current()
        if not self.ready and not self.calibrating:
            return
        if self.calibrating and display_only:
            self.missing(now, 'fresh hand landmarks')
            return
        observation = self._observation(pose, hand, segments, blocked, require_anchors=not self.ready)
        if observation is not None:
            observation['hand_side'] = hand_side
            angle_array = np.asarray(angles, float) if angles is not None else None
            observation['angles'] = (angle_array if angle_array is not None and angle_array.shape == (14,)
                                     and np.isfinite(angle_array).all() else None)
        if self.calibrating:
            if now is None:
                return
            if now < self.starts_at:
                self.status = f'Relax BODY grip; baseline in {min(3, math.ceil(self.starts_at - now))}s'
                return
            if observation is not None:
                if not self.samples or now > self.samples[-1][0]:
                    self.samples = [s for s in self.samples if now - s[0] <= 1.25]
                    if self.samples and now - self.samples[-1][0] > .5:
                        self.samples.clear()
                    self.samples.append((now, observation))
                # An occasional landmark miss/outlier must not throw away
                # every good sample. Require a recent, coherent group instead
                # of repeatedly starting from one arbitrary noisy first frame.
                same_hand = [s for s in self.samples if s[1].get('hand_side') == hand_side]
                if not same_hand:
                    self.stable_samples = 0
                    self.status = 'Waiting: fresh hand frame'
                    return
                center_hand = np.median([s[1]['hand'] for s in same_hand], axis=0)
                center_anchors = np.median([s[1]['anchors'] for s in same_hand], axis=0)
                coherent = []
                for sample in same_hand:
                    movement = sample[1]['hand'] - center_hand
                    palm_shift = np.linalg.norm(np.median(movement[[0, 5, 9, 13, 17]], axis=0))
                    anchor_shift = np.max(np.abs(center_anchors[:, :, 0] - sample[1]['anchors'][:, :, 0]))
                    if (palm_shift <= 5 and np.percentile(np.linalg.norm(movement, axis=1), 90) <= 10
                            and anchor_shift <= 8):
                        coherent.append(sample)
                self.stable_samples = len(coherent)
                if (len(coherent) >= 4 and now - coherent[0][0] >= .5
                        and coherent[-1][0] == now):
                    self.samples = coherent
                    self._finish(now)
                else:
                    self.status = f'Hold relaxed; samples {self.stable_samples}/4'
                    return
            else:
                self.missing(now, self.observation_reason)
                return
        if not self.ready:
            return
        if observation is None:
            self.status = 'Paused: ' + self.observation_reason
            return
        base = self.baseline
        if hand_side is not None and base['hand_side'] is not None and hand_side != base['hand_side']:
            self.status = 'Paused: hand identity changed; hold same grip'
            return
        current_hand, rest = observation['hand'], base['hand']
        palm, old_palm = current_hand[5] - current_hand[17], rest[5] - rest[17]
        scale = np.linalg.norm(palm) / np.linalg.norm(old_palm)
        agreement = (palm @ old_palm) / (np.linalg.norm(palm) * np.linalg.norm(old_palm))
        centers = [0, 5, 9, 13, 17]
        relocated = np.linalg.norm(np.mean(current_hand[centers] - rest[centers], axis=0)) > 40
        if not .65 <= scale <= 1.45 or agreement < .65 or relocated:
            self.status = 'Grip changed/rotated: relax and press H again'
            return
        rows = base['rows']
        curves = []
        for side in (0, 1):
            anchors = base['anchors'][side]
            line = np.interp(rows, anchors[:, 1], anchors[:, 0])
            valid = base['eligible']
            a, b = base['links'][side, valid].T
            u = base['weights'][side, valid]
            dx = ((1 - u) * (current_hand[a, 0] - rest[a, 0])
                  + u * (current_hand[b, 0] - rest[b, 0]))
            residual = dx - (line - base['reference'][side])[valid]
            # Endpoints retain the calibrated anchors in current bottle
            # coordinates. Fixed joint pairs carry their calibrated offsets.
            node_rows = np.r_[rows[0], rows[valid], rows[-1]]
            offsets = np.r_[0., residual, 0.]
            curves.append(line + np.interp(rows, node_rows, offsets))
        curves = np.asarray(curves)
        widths = curves[1] - curves[0]
        reference_widths = base['reference'][1] - base['reference'][0]
        valid = base['eligible'] & (widths > 20) & (widths < reference_widths * 1.8)
        if np.mean(valid[base['eligible']]) < .8:
            self.status = 'Estimate paused: incompatible/crossing joint contours'
            return
        covered = np.zeros(len(rows), bool)
        for i, y in enumerate(np.rint(rows).astype(int)):
            lo, hi = np.clip(np.rint(curves[:, i]).astype(int), 0, 319)
            if 0 <= y < 384 and hi > lo:
                covered[i] = np.any(observation['mask'][y, lo:hi + 1] > 0)
        valid &= covered
        if np.count_nonzero(valid) < 6:
            self.status = 'Estimate paused: keep the same body grip'
            return
        profiles = []
        for side in (0, 1):
            profile = np.interp(self.station_rows, rows, curves[side], left=np.nan, right=np.nan)
            eligible = np.interp(self.station_rows, rows, valid.astype(float), left=0, right=0) >= .99
            profile[~eligible] = np.nan
            profiles.append(profile)
            observed = observation['observed']
            near = ((np.abs(observed[:, 1, None] - rows[None, :]) <= 3)
                    & (np.abs(observed[:, 0, None] - curves[side][None, :]) <= 4))
            has_real_edge = np.any(near, axis=0)
            # Deliberately draw over an occluding hand, in a different color
            # and dashed style. Unobserved weak background is not filled in.
            visible_estimate = valid & ~has_real_edge
            for i in range(len(rows) - 1):
                if i % 5 < 3 and visible_estimate[i] and visible_estimate[i + 1]:
                    self.segments.append(pose.image_points([[curves[side, i], rows[i]],
                                                            [curves[side, i + 1], rows[i + 1]]]))
        both = np.isfinite(profiles[0]) & np.isfinite(profiles[1])
        if np.count_nonzero(both) < 3:
            self.segments.clear()
            self.status = 'Estimate paused: too little comparable grip region'
            return
        self.profile_left, self.profile_right = profiles
        self.anchor_source = 'calibrated_reference'
        self.display_only = display_only
        if display_only:
            self.profile_left[:] = self.profile_right[:] = np.nan
            self.status = 'Tracking only; hand detection recovering'
            return
        current_width = profiles[1][both] - profiles[0][both]
        reference = np.interp(self.station_rows[both], rows, reference_widths)
        self.metrics = dict(joint_estimated_width_px=float(np.median(current_width) * self.pixel_scale),
                            joint_estimated_width_reduction_pct=float(np.median(100 * (1 - current_width / reference))),
                            joint_estimated_profiles=int(np.count_nonzero(both)))
        self.status = 'Joint ESTIMATE; reference outline + hand'

    def context(self):
        result = {'method': 'fixed_joint_offsets_in_tracked_bottle_coordinates_v2',
                  'is_measured_bottle_width': False, 'ready': self.ready,
                  'pixel_units': 'initial_axis_scale_px',
                  'body_corridor_width_canonical_px': self.body_width,
                  'calibration_policy': 'wait_for_coherent_recent_samples_no_timeout',
                  'calibration_window_s': 1.25,
                  'calibration_min_samples': 4, 'calibration_min_span_s': .5}
        result['runtime_anchors'] = 'calibrated_reference_in_current_bottle_coordinates'
        if self.ready:
            result['reference_hand_side'] = self.baseline['hand_side']
            result['reference_angles_deg'] = self.baseline['angles'].tolist() if self.baseline['angles'] is not None else None
            for name in ('hand', 'anchors', 'rows', 'links', 'weights', 'reference', 'eligible'):
                result[name] = self.baseline[name].tolist()
        return result

    def draw(self, image):
        for segment in self.segments:
            first, last = np.rint(segment).astype(int)
            cv2.line(image, tuple(first), tuple(last), ESTIMATE_COLOR, 2, cv2.LINE_AA)
