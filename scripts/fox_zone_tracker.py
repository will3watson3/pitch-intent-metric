"""Recover a FOX overlay from independently agreeing views of the same pitch."""
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from detect_fox_zone import FoxSettings, detect_transparent_zone, boundary_profiles, profile_evidence, visible_transparent_zone
from refine_broadcast_zone import KEYS, box_edges, register_neighbor


class FoxZoneTracker:
    def __init__(self, root, video_directory=None, settings=None):
        self.root=Path(root)
        self.video_directory=Path(video_directory) if video_directory else None
        self.diagnostics=[]
        self.settings=settings or FoxSettings()
        detector_path=Path(__file__).with_name('detect_fox_zone.py')
        version=hashlib.sha256(detector_path.read_bytes()).hexdigest()[:12]
        self.cache_path=self.root/'.cache'/f'fox_candidates_{version}.json'
        self.cache=json.loads(self.cache_path.read_text()) if self.cache_path.is_file() else {}

    def video_frames(self, pitch_uid):
        if self.video_directory is None:
            return []
        path=self.video_directory/f'{pitch_uid}.mp4'
        if not path.is_file():
            return []
        capture=cv2.VideoCapture(str(path))
        frames=[]
        # Later frames supply geometry only, never catcher intent or glove position.
        for time in np.arange(2.8,6.01,.15):
            capture.set(cv2.CAP_PROP_POS_MSEC,float(time*1000))
            ok,image=capture.read()
            if not ok:
                break
            frames.append(dict(frame_uid=f'{pitch_uid}_zone_reference_{round(time*1000)}ms',
                               frame_time_sec=float(time),image=cv2.resize(image,(1280,720))))
        capture.release()
        return frames

    def fit(self, pitch_uid, target_row, rows, prior, usable_uids):
        target,candidates=self.prepare(pitch_uid,target_row,rows,prior,usable_uids)
        if target is None:
            return None,dict(reason='missing_target_image')
        return self.resolve(target,candidates)

    def prepare(self, pitch_uid, target_row, rows, prior, usable_uids):
        """Read and register observations once, independently of sweep settings."""
        target=cv2.imread(str(self.root/str(target_row.image_path)))
        if target is None:
            return None,[]
        sources=[]
        for _,row in rows.iterrows():
            if str(row.frame_uid) not in usable_uids:
                continue
            image=cv2.imread(str(self.root/str(row.image_path)))
            if image is not None:
                sources.append(dict(frame_uid=str(row.frame_uid),frame_time_sec=float(row.frame_time_sec),image=image))
        sources.extend(self.video_frames(pitch_uid))
        candidates=[]
        seen_times=set()
        for source in sources:
            time=round(source['frame_time_sec'], 3)
            if not np.isfinite(time) or time in seen_times:
                continue
            seen_times.add(time)
            fingerprint=hashlib.sha256(source['image'].tobytes()).hexdigest()[:16]
            key=json.dumps([pitch_uid,source['frame_uid'],fingerprint,[float(prior.get(k,np.nan)) for k in KEYS],self.settings.centered_search])
            if key not in self.cache:
                self.cache[key]=detect_transparent_zone(source['image'],prior,centered_search=self.settings.centered_search)
            fit=self.cache[key]
            if fit is None:
                continue
            self.diagnostics.append(dict(pitch_uid=pitch_uid,frame_uid=source['frame_uid'],**fit))
            if fit['fox_score'] < .20 or fit['fox_min_support'] < 0:
                continue
            profiles=boundary_profiles(source['image'],fit)
            if profiles is None:
                continue
            if source['frame_uid']==str(target_row.frame_uid):
                transformed={key:fit[key] for key in KEYS}
            else:
                cv2.setRNGSeed(0)
                transformed=register_neighbor(source['image'],target,fit)
            if transformed is None:
                continue
            candidates.append(dict(zone=transformed,fit=fit,profiles=profiles,
                                   frame_uid=source['frame_uid'],time=time))
        self.cache_path.parent.mkdir(parents=True,exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache))
        return target,candidates

    def resolve(self, target, candidates, settings=None):
        settings=settings or self.settings
        candidates=[c for c in candidates if c['fit']['fox_score']>=settings.min_score
                    and c['fit']['fox_min_support']>=settings.min_side]
        if len(candidates)<settings.min_frames:
            return None,dict(reason='insufficient_registered_fox_evidence',candidate_count=len(candidates))
        edges=np.stack([box_edges(c['zone']) for c in candidates])
        # One extra pixel covers interpolation from subpixel camera registration;
        # temporal support and the independent holdout still guard false boxes.
        tolerance=settings.geometry_tolerance*target.shape[1]/1280
        distance=np.max(np.abs(edges[:,None]-edges[None,:]),axis=2)
        times=np.array([c['time'] for c in candidates])
        # A single nearby window must support the rectangle, not scattered frames
        # across camera cuts or separate appearances of the graphic.
        clusters=[np.flatnonzero((row<=tolerance)&(times>=times[i])&(times<=times[i]+settings.window_seconds))
                  for i,row in enumerate(distance)]
        ranked=sorted(clusters,key=lambda group:(len(group),sum(candidates[i]['fit']['fox_score'] for i in group)),reverse=True)
        seen=set()
        failures=[]
        for best in ranked:
            signature=tuple(np.round(np.median(edges[best],axis=0)/2).astype(int))
            if signature in seen:
                continue
            seen.add(signature)
            zone,detail=self.resolve_cluster(target,candidates,edges,best,settings,tolerance)
            if zone is not None:
                return zone,detail
            failures.append(detail)
            if len(seen)>=settings.max_clusters:
                break
        return None,failures[0]

    def resolve_cluster(self,target,candidates,edges,best,settings,tolerance):
        median=np.median(edges[best],axis=0)
        best=best[np.max(abs(edges[best]-median),axis=1)<=tolerance]
        if len(best)<settings.min_frames:
            return None,dict(reason='unstable_fox_geometry',candidate_count=len(candidates),support=len(best))
        kept=[candidates[i] for i in best]
        if max(c['time'] for c in kept)-min(c['time'] for c in kept)<.35:
            return None,dict(reason='insufficient_fox_time_span')
        # Alternating observations must recover the same rectangle independently.
        ordered=sorted(kept,key=lambda c:c['time'])
        a=np.median([box_edges(c['zone']) for c in ordered[::2]],axis=0)
        b=np.median([box_edges(c['zone']) for c in ordered[1::2]],axis=0)
        if np.max(abs(a-b))>settings.split_tolerance*target.shape[1]/1280:
            return None,dict(reason='fox_temporal_holdout_disagreement')
        zone={key:float(np.median([c['zone'][key] for c in kept])) for key in KEYS}
        temporal=profile_evidence(np.median([c['profiles'] for c in kept],axis=0),settings)
        if not temporal['accepted']:
            return None,dict(reason='weak_fox_temporal_contrast',**temporal)
        visible=visible_transparent_zone(target,zone,settings)
        if not visible['accepted']:
            return None,{**visible,'reason':visible.get('reason','fox_overlay_too_faint_in_target')}
        # The temporal median establishes that the graphic is real. When the
        # target frame independently locates the same box, its own edges avoid
        # a small camera-registration bias in the transported median.
        zone=visible['local_zone']
        return zone,dict(reason='fox_temporal_alpha_fit',fox_frame_support=len(kept),
                         fox_visible_fraction=visible['fox_visible_fraction'],
                         fox_contrast_px=visible['fox_contrast_px'],
                         fox_temporal_visible_fraction=temporal['fox_visible_fraction'],
                         fox_candidate_count=len(candidates),fox_temporal_error_px=float(np.max(abs(a-b))),
                         fox_score=float(np.median([c['fit']['fox_score'] for c in kept])),
                         fox_reference_frames=';'.join(c['frame_uid'] for c in kept))
