"""Separate detector for translucent, borderless broadcast rectangles.

This module never calls or changes the four-stroke fitter. Positive inward
brightness steps, finite side lengths, and independent temporal agreement
provide evidence for an alpha-composited rectangle.
"""
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class FoxSettings:
    min_score: float = .30
    min_side: float = .12
    min_frames: int = 5
    window_seconds: float = 1.
    geometry_tolerance: float = 5.
    split_tolerance: float = 4.
    min_fraction: float = .15
    mean_fraction: float = .20
    local_tolerance: float = 4.
    constrained_local: bool = False
    max_clusters: int = 1
    centered_search: bool = False


def boundary_profiles(image, zone):
    """Sample both sides of each boundary in native 720p pixel distances."""
    if image is None:
        return None
    try:
        x, y, w, h = [float(zone[k]) for k in ('zone_x', 'zone_y', 'zone_width', 'zone_height')]
    except (KeyError, TypeError, ValueError):
        return None
    ih, iw = image.shape[:2]
    x, w = x*1280/iw, w*1280/iw
    y, h = y*720/ih, h*720/ih
    if not np.isfinite([x, y, w, h]).all() or not (40 <= w <= 110 and 50 <= h <= 145):
        return None
    l, r, t, b = x-w/2, x+w/2, y-h/2, y+h/2
    if l < 12 or r > 1267 or t < 12 or b > 707:
        return None
    small = cv2.resize(image, (1280, 720)) if (iw, ih) != (1280, 720) else image
    small = cv2.GaussianBlur(small.astype(np.float32), (3, 3), .65)
    along = np.linspace(.12, .88, 64, dtype=np.float32)[:, None]
    distance = np.array([-9, -6, -3, 3], dtype=np.float32)[None, :]
    coordinates = [(l+distance, t+along*h), (r-distance, t+along*h),
                   (l+along*w, t+distance), (l+along*w, b-distance)]
    return np.stack([cv2.remap(small, *[a.astype(np.float32) for a in np.broadcast_arrays(xx, yy)],
                               cv2.INTER_LINEAR) for xx, yy in coordinates])


def profile_evidence(profiles, settings=None):
    """Compare interior tint with a locally extrapolated exterior background.

    For a sequence, the caller takes the temporal median of aligned profiles.
    This suppresses moving occluders; it does not invent an overlay-free frame.
    Absolute contrast is required so normalization cannot amplify invisible tint.
    """
    settings = settings or FoxSettings()
    if profiles is None:
        return dict(accepted=False, fox_visible_fraction=0., fox_contrast_px=0.)
    outside = profiles[:, :, :3]
    # Do not extrapolate a jersey/grass edge into a huge fictitious brightness
    # change. The background model represents slow lighting variation only.
    slope = np.clip((outside[:, :, 2]-outside[:, :, 0])/6, -1., 1.)
    background = outside.mean(axis=2)+9*slope
    inside = profiles[:, :, 3]
    delta = inside-background
    alpha = delta/np.maximum(255-background, 20)
    contrast = np.median(delta, axis=2)
    tint = np.median(alpha, axis=2)
    # Curved/textured backgrounds are poor predictors of the underlying color.
    curvature = np.median(abs(outside[:, :, 0]-2*outside[:, :, 1]+outside[:, :, 2]), axis=2)
    support = ((contrast >= 2.5) & (tint >= .018) & (tint <= .24)
               & (np.std(alpha, axis=2) <= .08) & (curvature <= 12))
    fractions = support.mean(axis=1)
    # A catcher frequently masks one segment of a real translucent boundary.
    # Require support on every side, but let the temporal median fill a short
    # occlusion instead of rejecting the whole box.
    return dict(accepted=bool(fractions.min() >= settings.min_fraction and fractions.mean() >= settings.mean_fraction),
                fox_visible_fraction=float(fractions.min()),
                fox_contrast_px=float(np.median(contrast)),
                fox_boundary_fractions=','.join(f'{v:.3f}' for v in fractions))


def visible_transparent_zone(image, zone, settings=None):
    settings = settings or FoxSettings()
    local = None
    if settings.constrained_local:
        local = detect_transparent_zone(image, zone, max_edge_shift=settings.local_tolerance, centered_search=settings.centered_search)
        if local is None or not local['accepted']:
            return dict(accepted=False, reason='no_local_fox_boundaries')
    evidence = profile_evidence(boundary_profiles(image, local or zone), settings)
    if not evidence['accepted']:
        return evidence
    # A transferred rectangle must also match independently detected local
    # boundaries. Tint somewhere along a player's outline is insufficient.
    local = local or detect_transparent_zone(image, zone, centered_search=settings.centered_search)
    if local is None or not local['accepted']:
        return {**evidence, 'accepted': False}
    def edges(box):
        x,y,w,h = [float(box[k]) for k in ('zone_x','zone_y','zone_width','zone_height')]
        return np.array([x-w/2,y-h/2,x+w/2,y+h/2])
    error = np.max(abs(edges(local)-edges(zone))/np.array([image.shape[1]/1280,image.shape[0]/720]*2))
    return {**evidence, 'accepted': bool(error <= settings.local_tolerance),
            'fox_local_error_px': float(error),
            'local_zone': {k:local[k] for k in ('zone_x','zone_y','zone_width','zone_height')}}


def step_maps(image):
    rgb = cv2.GaussianBlur(image.astype(np.float32), (3, 3), .65)
    def evidence(inside, outside):
        alpha = (inside-outside)/(275-outside)
        common = np.median(alpha, axis=2)
        disagreement = np.std(alpha, axis=2)
        baseline = np.exp(-.5*(.07/.035)**2)
        response = (np.exp(-.5*((common-.07)/.035)**2)-baseline)/(1-baseline)
        return response*np.exp(-disagreement/.045)
    xp = sum(np.roll(rgb, -d, 1) for d in (1, 2, 3))/3
    xm = sum(np.roll(rgb, d, 1) for d in (1, 2, 3))/3
    yp = sum(np.roll(rgb, -d, 0) for d in (1, 2, 3))/3
    ym = sum(np.roll(rgb, d, 0) for d in (1, 2, 3))/3
    return evidence(xp,xm), evidence(xm,xp), evidence(yp,ym), evidence(ym,yp)


def detect_transparent_zone(image, prior=None, maps=None, max_edge_shift=None, centered_search=False):
    """Return a candidate and evidence, not a calibrated accuracy probability."""
    if image is None:
        return None
    height, width = image.shape[:2]
    # Native 1280x720 coordinates make contrast kernels resolution-independent.
    small = cv2.resize(image, (1280, 720)) if (width, height) != (1280, 720) else image
    x0, x1, y0, y1 = 470, 800, 170, 425
    px, py, pw, ph = 640., 300., 65., 85.
    if prior is not None:
        values = [prior.get(k, np.nan) for k in ('zone_x','zone_y','zone_width','zone_height')]
        if np.isfinite(np.asarray(values, dtype=float)).all() and .36*width <= float(values[0]) <= .62*width and .24*height <= float(values[1]) <= .54*height:
            px, py, pw, ph = map(float, values)
            px *= 1280/width; pw *= 1280/width
            py *= 720/height; ph *= 720/height
    # The old candidate limits stop at x=780 (right) and x=485 (left).
    # Recenter when even the supplied RF box falls outside those limits. This
    # changes only clipped searches; supplied evidence maps retain their origin.
    clipped_prior=px+pw/2>x1-20 or px-pw/2<x0+15
    if centered_search or (maps is None and clipped_prior):
        x0=int(np.clip(round(px-165),200,750))
        x1=x0+330
    crop = small[y0:y1, x0:x1]
    left, right, top, bottom = step_maps(crop) if maps is None else maps
    def peaks(profile, lo, hi, count=24):
        choices=[]
        for index in np.argsort(profile)[::-1]:
            if lo <= index <= hi and all(abs(index-existing)>2 for existing in choices):
                choices.append(int(index))
                if len(choices) == count:
                    break
        return np.array(choices, dtype=int)
    # Search broadly enough to recover RF's frequent catcher/umpire confusion.
    yt=max(12,int(py-y0-65)); yb=min(crop.shape[0]-12,int(py-y0+65))
    xl=max(12,int(px-x0-85)); xr=min(crop.shape[1]-12,int(px-x0+85))
    ls=peaks(left[yt:yb].mean(axis=0),max(15,px-x0-115),min(235,px-x0+40))
    rs=peaks(right[yt:yb].mean(axis=0),max(50,px-x0-35),min(310,px-x0+115))
    ts=peaks(top[:,xl:xr].mean(axis=1),max(15,py-y0-105),min(170,py-y0+5))
    bs=peaks(bottom[:,xl:xr].mean(axis=1),max(70,py-y0-10),min(235,py-y0+100))
    if max_edge_shift is not None:
        # Refine each established temporal edge locally. A broad re-search can
        # select a different rectangle on the catcher's uniform instead.
        if prior is None or not np.isfinite([px,py,pw,ph]).all():
            return None
        def near(center, limit):
            return np.arange(max(12,int(np.ceil(center-max_edge_shift))),
                             min(limit-13,int(np.floor(center+max_edge_shift)))+1,dtype=int)
        ls,rs=near(px-x0-pw/2,crop.shape[1]),near(px-x0+pw/2,crop.shape[1])
        ts,bs=near(py-y0-ph/2,crop.shape[0]),near(py-y0+ph/2,crop.shape[0])
    if any(len(a)==0 for a in (ls,rs,ts,bs)):
        return None
    l,r,t,b=np.meshgrid(ls,rs,ts,bs,indexing='ij')
    l,r,t,b=[a.ravel() for a in (l,r,t,b)]
    w,h=r-l,b-t
    valid=(w>=45)&(w<=105)&(h>=55)&(h<=135)&(h>=1.10*w)&(h<=1.65*w)
    l,r,t,b,w,h=[a[valid] for a in (l,r,t,b,w,h)]
    if not len(l):
        return None
    vi=[np.pad(np.cumsum(m,axis=0),((1,0),(0,0))) for m in (left,right)]
    hi=[np.pad(np.cumsum(m,axis=1),((0,0),(1,0))) for m in (top,bottom)]
    scores=np.stack([(vi[0][b-3,l]-vi[0][t+4,l])/(h-7),
                     (vi[1][b-3,r]-vi[1][t+4,r])/(h-7),
                     (hi[0][t,r-3]-hi[0][t,l+4])/(w-7),
                     (hi[1][b,r-3]-hi[1][b,l+4])/(w-7)],axis=1)
    extensions=np.stack([(vi[0][t-3,l]-vi[0][t-12,l]+vi[0][b+12,l]-vi[0][b+3,l])/18,
                         (vi[1][t-3,r]-vi[1][t-12,r]+vi[1][b+12,r]-vi[1][b+3,r])/18,
                         (hi[0][t,l-3]-hi[0][t,l-12]+hi[0][t,r+12]-hi[0][t,r+3])/18,
                         (hi[1][b,l-3]-hi[1][b,l-12]+hi[1][b,r+12]-hi[1][b,r+3])/18],axis=1)
    def vertical_end(integral, x, y, inward):
        if inward == 1:
            return (integral[y+12,x]-integral[y+2,x]-integral[y-2,x]+integral[y-12,x])/10
        return (integral[y-2,x]-integral[y-12,x]-integral[y+12,x]+integral[y+2,x])/10
    def horizontal_end(integral, y, x, inward):
        if inward == 1:
            return (integral[y,x+12]-integral[y,x+2]-integral[y,x-2]+integral[y,x-12])/10
        return (integral[y,x-2]-integral[y,x-12]-integral[y,x+12]+integral[y,x+2])/10
    corners=np.stack([
        np.maximum(vertical_end(vi[0],l,t,1),horizontal_end(hi[0],t,l,1)),
        np.maximum(vertical_end(vi[1],r,t,1),horizontal_end(hi[0],t,r,-1)),
        np.maximum(vertical_end(vi[0],l,b,-1),horizontal_end(hi[1],b,l,1)),
        np.maximum(vertical_end(vi[1],r,b,-1),horizontal_end(hi[1],b,r,-1)),
    ],axis=1)
    scores=scores-.25*np.maximum(extensions,0)
    penalty=.025*(abs((l+r)/2+x0-px)+abs((t+b)/2+y0-py))/60
    score=.50*scores.mean(axis=1)+.15*scores.min(axis=1)+.35*corners.mean(axis=1)-penalty
    index=int(np.argmax(score))
    return dict(zone_x=float((l[index]+r[index])/2+x0)*width/1280,
                zone_y=float((t[index]+b[index])/2+y0)*height/720,
                zone_width=float(w[index])*width/1280,zone_height=float(h[index])*height/720,
                fox_score=float(score[index]),fox_min_support=float(scores[index].min()),
                fox_side_support=','.join(f'{s:.3f}' for s in scores[index]),fox_corner_score=float(corners[index].mean()),
                accepted=bool(score[index]>=.30 and scores[index].min()>=.12))


class FoxBroadcastClassifier:
    """Match FOX/FS1 logo artwork; geography/game IDs are never classifier inputs."""
    def __init__(self, template_directory):
        from pathlib import Path
        self.templates=[]
        for path in sorted(Path(template_directory).glob('*.png')):
            template=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if template is not None:
                self.templates.append((path.stem,template))

    def classify(self, image):
        if image is None or not self.templates:
            return dict(is_fox=False,score=0.,logo='')
        small=cv2.resize(image,(1280,720))
        region=cv2.cvtColor(small[:125,950:],cv2.COLOR_BGR2GRAY)
        best=(0.,'')
        for name,template in self.templates:
            for scale in (.8,1.,1.2):
                scaled=cv2.resize(template,None,fx=scale,fy=scale)
                score=float(cv2.matchTemplate(region,scaled,cv2.TM_CCOEFF_NORMED).max())
                if score>best[0]:best=(score,name)
        return dict(is_fox=best[0]>=.72,score=best[0],logo=best[1])
