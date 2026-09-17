"""Locate the visible broadcast rectangle using RF geometry as the search prior.

Scores all four thin, bright strokes jointly; a player's silhouette or a pair
of unrelated vertical lines is insufficient evidence for a correction.
"""
from pathlib import Path
import cv2
import numpy as np

KEYS = ('zone_x', 'zone_y', 'zone_width', 'zone_height')


def finite_box(box, shape):
    try:
        x, y, w, h = (float(box[k]) for k in KEYS)
    except (KeyError, TypeError, ValueError):
        return False
    ih, iw = shape[:2]
    return bool(np.isfinite([x,y,w,h]).all() and 20*iw/1280 <= w <= 150*iw/1280
                and 25*ih/720 <= h <= 180*ih/720 and x-w/2 >= 0
                and y-h/2 >= 0 and x+w/2 < iw and y+h/2 < ih)


def refine_zone(image, baseline):
    if image is None or not finite_box(baseline, image.shape):
        return None
    x,y,w,h = (float(baseline[k]) for k in KEYS)
    scale = image.shape[1]/1280
    radius = max(12, int(24*scale))
    x0 = max(0, int(x-w/2-radius-5)); x1 = min(image.shape[1], int(x+w/2+radius+6))
    y0 = max(0, int(y-h/2-radius-5)); y1 = min(image.shape[0], int(y+h/2+radius+6))
    gray = cv2.cvtColor(image[y0:y1,x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Symmetric contrast rejects ordinary object boundaries and broad white fabric.
    def ridge(axis):
        responses=[]
        for d in (2,3,4):
            d=max(1,round(d*scale))
            responses.append(np.minimum(gray-np.roll(gray,d,axis),gray-np.roll(gray,-d,axis)))
        r=np.maximum.reduce(responses)
        return np.clip(r/28,0,1) * np.clip((gray-45)/60,0,1)
    vr=ridge(1); hr=ridge(0)
    # A one-pixel tolerance handles antialiasing while retaining precise coordinates.
    v=cv2.dilate(vr,np.ones((1,3),np.uint8)); q=cv2.dilate(hr,np.ones((3,1),np.uint8))
    def candidates(center, profile):
        lo=max(4,int(center-radius)); hi=min(len(profile)-4,int(center+radius)+1)
        ids=np.arange(lo,hi)
        order=ids[np.argsort(profile[ids])[::-1]]
        picked=[]
        for n in order:
            if all(abs(n-m)>2 for m in picked): picked.append(int(n))
            if len(picked)==10: break
        return np.array(picked,dtype=int)
    yt=max(0,int(y-y0-h*.35));yb=min(len(gray),int(y-y0+h*.35))
    xl=max(0,int(x-x0-w*.35));xr=min(gray.shape[1],int(x-x0+w*.35))
    ls=candidates(x-x0-w/2,vr[yt:yb].mean(axis=0));rs=candidates(x-x0+w/2,vr[yt:yb].mean(axis=0))
    ts=candidates(y-y0-h/2,hr[:,xl:xr].mean(axis=1));bs=candidates(y-y0+h/2,hr[:,xl:xr].mean(axis=1))
    if any(len(a)==0 for a in (ls,rs,ts,bs)):return None
    l,r,t,b=np.meshgrid(ls,rs,ts,bs,indexing='ij');l=l.ravel();r=r.ravel();t=t.ravel();b=b.ravel()
    widths=r-l;heights=b-t
    valid=(widths>=.60*w)&(widths<=1.45*w)&(heights>=.60*h)&(heights<=1.45*h)&(heights>=.75*widths)&(heights<=2.2*widths)
    l,r,t,b,widths,heights=[a[valid] for a in (l,r,t,b,widths,heights)]
    if len(l)==0:return None
    vi=np.pad(np.cumsum(v,axis=0),((1,0),(0,0)));hi=np.pad(np.cumsum(q,axis=1),((0,0),(1,0)))
    sides=np.stack([(vi[b-2,l]-vi[t+3,l])/(heights-5),(vi[b-2,r]-vi[t+3,r])/(heights-5),
                    (hi[t,r-2]-hi[t,l+3])/(widths-5),(hi[b,r-2]-hi[b,l+3])/(widths-5)],axis=1)
    prior=(abs((l+r)/2+x0-x)+abs((t+b)/2+y0-y)+.5*abs(widths-w)+.5*abs(heights-h))/(4*radius)
    score=.65*sides.mean(axis=1)+.35*sides.min(axis=1)-.08*prior
    i=int(np.argmax(score))
    # Refine within the antialiasing tolerance to the ridge center, not its dilation.
    def peak_line(arr,pos,start,end,axis):
        ids=np.arange(max(0,pos-1), min(arr.shape[axis],pos+2))
        vals=np.array([arr[start:end,n].mean() if axis==1 else arr[n,start:end].mean() for n in ids])
        return float(ids[np.argmax(vals)])
    ll=peak_line(vr,l[i],t[i]+3,b[i]-2,1)+x0;rr=peak_line(vr,r[i],t[i]+3,b[i]-2,1)+x0
    tt=peak_line(hr,t[i],l[i]+3,r[i]-2,0)+y0;bb=peak_line(hr,b[i],l[i]+3,r[i]-2,0)+y0
    return dict(zone_x=(ll+rr)/2,zone_y=(tt+bb)/2,zone_width=rr-ll,zone_height=bb-tt,
                edge_score=float(score[i]),edge_min_support=float(sides[i].min()),
                edge_support=','.join(f'{a:.3f}' for a in sides[i]),
                accepted=bool(score[i]>=.72 and sides[i].min()>=.55))


def box_edges(box):
    x,y,w,h=(float(box[k]) for k in KEYS)
    return np.array([x-w/2,y-h/2,x+w/2,y+h/2])


def register_neighbor(source, target, box):
    """Transport nearby evidence only when background features prove camera alignment."""
    if source is None or target is None or source.shape != target.shape:
        return None
    a=cv2.cvtColor(source,cv2.COLOR_BGR2GRAY);b=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
    mask=np.zeros_like(a); ih,iw=a.shape
    x,y,w,h=(float(box[k]) for k in KEYS)
    cv2.rectangle(mask,(max(0,int(x-220)),max(0,int(y-160))),
                  (min(iw-1,int(x+220)),min(ih-1,int(y+160))),255,-1)
    # Exclude the catcher and the graphic: moving people cannot establish camera motion.
    cv2.rectangle(mask,(int(x-w-25),int(y-h-25)),(int(x+w+25),int(y+h+25)),0,-1)
    points=cv2.goodFeaturesToTrack(a,250,.02,8,mask=mask)
    if points is None or len(points)<15:return None
    moved,ok,_=cv2.calcOpticalFlowPyrLK(a,b,points,None)
    if moved is None:return None
    back,back_ok,_=cv2.calcOpticalFlowPyrLK(b,a,moved,None)
    if back is None:return None
    valid=(ok.ravel()>0)&(back_ok.ravel()>0)&(np.linalg.norm(points-back,axis=2).ravel()<1)
    if valid.sum()<15:return None
    matrix,inliers=cv2.estimateAffinePartial2D(points[valid],moved[valid],method=cv2.RANSAC,ransacReprojThreshold=1.5)
    if matrix is None or inliers.sum()<15 or inliers.mean()<.65:return None
    scale=float(np.hypot(matrix[0,0],matrix[1,0]));angle=float(np.degrees(np.arctan2(matrix[1,0],matrix[0,0])))
    if not .97<=scale<=1.03 or abs(angle)>.4 or np.linalg.norm(matrix[:,2])>25:return None
    center=matrix@np.array([x,y,1.]);return dict(zone_x=center[0],zone_y=center[1],zone_width=w*scale,zone_height=h*scale)


class ZoneRefiner:
    def __init__(self, dense, root):
        self.root=Path(root)
        self.by_pitch={str(uid): rows for uid,rows in dense.groupby('pitch_uid')}

    def image(self, path):
        path=self.root/str(path)
        return cv2.imread(str(path)) if path.is_file() else None

    def resolve(self, pitch_uid, frame_uid, image_path, fallback):
        rows=self.by_pitch.get(str(pitch_uid))
        image=self.image(image_path)
        metadata=dict(zone_refinement_status='unverified_fallback',zone_edge_score='',
                      zone_edge_min_support='',zone_refinement_frame_support=0,
                      zone_refinement_reference_frame_uid='',zone_baseline_source='legacy_fallback')
        if image is None:
            metadata['zone_refinement_status']='missing_image'
            return fallback,metadata
        priors=[];selected=None
        if rows is not None:
            matches=rows[rows.frame_uid.astype(str)==str(frame_uid)]
            if not matches.empty:
                selected=matches.iloc[-1]
                if finite_box(selected,image.shape):priors.append(('roboflow_selected_frame',selected))
            good=rows[rows.apply(lambda r: finite_box(r,image.shape),axis=1)]
            if not good.empty:
                priors.append(('roboflow_pitch_median',good[list(KEYS)].median().to_dict()))
        if fallback and finite_box(fallback,image.shape):priors.append(('legacy_fallback',fallback))
        chosen=None
        for source,prior in priors:
            if selected is None or not finite_box(selected, image.shape):
                continue
            fitted=refine_zone(image,prior)
            if fitted and fitted['accepted']:
                chosen=fitted;metadata['zone_baseline_source']=source;break
        if chosen:
            metadata.update(zone_refinement_status='visible_four_edge_fit',zone_edge_score=chosen['edge_score'],
                            zone_edge_min_support=chosen['edge_min_support'],zone_refinement_frame_support=1,
                            zone_refinement_reference_frame_uid=str(frame_uid))
            return {k:chosen[k] for k in KEYS},metadata
        # No visible rectangle: require multiple nearby, camera-registered observations.
        neighbors=[]
        if selected is not None and 'frame_time_sec' in rows:
            try:time=float(selected.frame_time_sec)
            except (TypeError,ValueError):time=np.nan
            near=rows.assign(distance=(rows.frame_time_sec.astype(float)-time).abs())
            near=near[(near.distance>0)&(near.distance<=.35)].sort_values('distance').head(8)
            for _,row in near.iterrows():
                other=self.image(row.image_path)
                if other is None:continue
                fit=refine_zone(other,row)
                if not fit or not fit['accepted']:continue
                registered=register_neighbor(other,image,fit)
                if registered:
                    neighbors.append((registered,fit,str(row.frame_uid)))
        if len(neighbors)>=3:
            edges=np.stack([box_edges(z[0]) for z in neighbors]); med=np.median(edges,axis=0)
            inliers=np.max(abs(edges-med),axis=1)<=3*image.shape[1]/1280
            if inliers.sum()>=3:
                kept=[n for n,keep in zip(neighbors,inliers) if keep]
                z={k:float(np.median([n[0][k] for n in kept])) for k in KEYS}
                metadata.update(zone_refinement_status='registered_temporal_fit',zone_edge_score=float(np.median([n[1]['edge_score'] for n in kept])),
                                zone_edge_min_support=float(np.median([n[1]['edge_min_support'] for n in kept])),zone_refinement_frame_support=len(kept),
                                zone_refinement_reference_frame_uid=';'.join(n[2] for n in kept),zone_baseline_source='roboflow_neighbor_frames')
                return z,metadata
        if selected is not None and not finite_box(selected, image.shape):
            metadata["zone_refinement_status"] = "no_zone_evidence"
            return None, metadata
        # Preserve a local RF estimate instead of assigning another game's camera geometry.
        if priors:
            source,prior=priors[0];metadata['zone_baseline_source']=source
            return {k:float(prior[k]) for k in KEYS},metadata
        return None,metadata
