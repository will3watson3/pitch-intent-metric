"""Reject close-ups, crowd shots, and camera transitions before zone recovery."""
import cv2
import numpy as np


def pitching_view(image, pose=None, detection=None):
    if image is None:
        return dict(usable=False, reason='missing_image', score=0.)
    pose = {} if pose is None else pose
    detection = {} if detection is None else detection
    def number(key):
        try:
            return float(pose.get(key, np.nan))
        except (TypeError, ValueError):
            return np.nan
    visibility=number('pose_visibility')
    center_x,center_y=number('pose_center_x'),number('pose_center_y')
    torso=number('torso_length')
    pose_good=(visibility>=.60 and .12<=center_x<=.68 and .42<=center_y<=.86 and .045<=torso<=.22)
    # A confident Roboflow rectangle far from the plate-center search region is
    # characteristic of the pitcher/field camera.  This is a view gate only;
    # the rectangle is never accepted as recovered geometry here.
    try:
        zone_confidence=float(detection.get('zone_confidence',np.nan))
        zone_x=float(detection.get('zone_x',np.nan))/image.shape[1]
        zone_y=float(detection.get('zone_y',np.nan))/image.shape[0]
        zone_width=float(detection.get('zone_width',np.nan))/image.shape[1]
        zone_height=float(detection.get('zone_height',np.nan))/image.shape[0]
    except (TypeError,ValueError,ZeroDivisionError):
        zone_confidence=zone_x=zone_y=zone_width=zone_height=np.nan
    if zone_confidence>=.30 and np.isfinite(zone_x) and not .34<=zone_x<=.66:
        return dict(usable=False,reason='off_center_non_plate_camera',score=0.)
    centered_plate_detection=(zone_confidence>=.55 and .34<=zone_x<=.66 and .20<=zone_y<=.60
                              and .025<=zone_width<=.12 and .05<=zone_height<=.20)
    if not pose_good and not centered_plate_detection:
        return dict(usable=False, reason='no_pitcher_or_plate_view_evidence', score=0.)
    if not pose_good:
        return dict(usable=True,reason='strong_centered_plate_detection',score=float(zone_confidence))
    reason='pitcher_pose_and_centered_plate' if centered_plate_detection else 'pitcher_pose_and_scale'
    return dict(usable=True, reason=reason, score=float(visibility))


def stable_view_mask(images, poses, detections=None):
    """Require a run of at least three usable frames; cuts split candidate runs."""
    detections=[{} for _ in images] if detections is None else detections
    results=[pitching_view(image,pose,detection) for image,pose,detection in zip(images,poses,detections)]
    flags=np.array([r['usable'] for r in results])
    start=0
    for end in range(len(flags)+1):
        if end==len(flags) or not flags[end]:
            run=results[start:end]
            minimum=2 if run and all('centered_plate' in r['reason'] for r in run) else 3
            if end-start<minimum:
                for index in range(start,end):
                    results[index]=dict(usable=False,reason='camera_transition_or_short_view',score=0.)
            start=end+1
    return results
