"""Opt-in second pass for FOX overlays and unusable setup frames.

Existing successful four-edge/registered rows are copied without alteration.
A failure clears geometry instead of allowing an unsupported RF guess through.
"""
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from detect_fox_zone import FoxBroadcastClassifier, visible_transparent_zone
from detect_setup_frame import stable_view_mask
from fox_zone_tracker import FoxZoneTracker
from refine_broadcast_zone import KEYS, refine_zone, register_neighbor
from apply_calibrated_zones_to_predictions import recompute_target
import select_best_glove_frame as glove_selector

ROOT=Path(__file__).resolve().parents[1]
PROTECTED={'visible_four_edge_fit','registered_temporal_fit','manual_frame_override','recovered_four_edge_frame'}
RECOVERED={'fox_temporal_alpha_fit','fox_at_bat_consensus_fit','recovered_four_edge_frame'}


def write_records(path, records, default_fields=None):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in records for key in row)) or list(default_fields or [])
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


class RecoveryPipeline:
    def __init__(self, dense, poses, video_directory):
        self.rows_by_pitch={str(uid):rows.sort_values('frame_time_sec') for uid,rows in dense.groupby('pitch_uid')}
        self.poses=poses.drop_duplicates('frame_uid').set_index('frame_uid')
        self.classifier=FoxBroadcastClassifier(ROOT/'models/fox_zone_logos')
        self.tracker=FoxZoneTracker(ROOT,video_directory)
        self.frame_records=[]
        self.audit=[]
        self.game_priors={}
        self.args=glove_selector.parse_args(['--motion-aware','--raw-glove-candidates','--allow-glove-only-frames',
            '--window-start-sec','1.2','--window-end-sec','2.7','--min-target-x','-0.75',
            '--max-target-x','1.65','--min-target-y','-0.60','--max-target-y','1.35','--probable-mask-target-y','1.35'])

    def reject(self, base, reason):
        record=dict(base)
        for key in [*KEYS,'vision_target_x_01','vision_target_y_01']:
            record[key]=''
        for key in list(record):
            if key.startswith('fox_') or key in {'zone_refinement_frame_support','zone_refinement_reference_frame_uid',
                                                'zone_edge_score','zone_edge_min_support'}:
                record[key]=''
        record.update(zone_confidence=0,zone_refinement_status='recovery_rejected',
                      zone_source='missing_broadcast_rectangle',zone_registration_source='recovery_rejected',
                      status='missing_required_detection',error=reason,recovery_reason=reason,
                      recovery_original_frame_uid=base.get('frame_uid',''),recovery_frame_changed='no')
        return record

    def recover(self, base):
        if base.get('zone_refinement_status') in PROTECTED:
            return dict(base)
        uid=str(base['pitch_uid'])
        rows=self.rows_by_pitch.get(uid)
        if rows is None:
            return self.reject(base,'missing_clip_detections')
        images={str(row.frame_uid):cv2.imread(str(ROOT/str(row.image_path))) for _,row in rows.iterrows()}
        pose_rows=[self.poses.loc[str(row.frame_uid)] if str(row.frame_uid) in self.poses.index else {} for _,row in rows.iterrows()]
        view_results=stable_view_mask(list(images.values()),pose_rows,[row for _,row in rows.iterrows()])
        usable={str(row.frame_uid) for (_,row),view in zip(rows.iterrows(),view_results) if view['usable']}
        frame_lookup={str(row.frame_uid):row for _,row in rows.iterrows()}
        selected_uid=base.get('selected_frame_uid') or base['frame_uid']
        selected=frame_lookup.get(selected_uid)
        if not usable:
            for _,row in rows.iterrows():
                self.frame_records.append(dict(frame_uid=row.frame_uid,pitch_uid=uid,zone_refinement_status='recovery_rejected',frame_view_status='no_usable_pitching_view',**{key:'' for key in KEYS}))
            return self.reject(base,'no_usable_pitching_view')
        reference=selected if selected_uid in usable else next(row for _,row in rows.iterrows() if str(row.frame_uid) in usable)
        broadcast=self.classifier.classify(images[str(reference.frame_uid)])
        zones={}
        evidence={}
        if broadcast['is_fox']:
            prior_row=reference
            prior={key:glove_selector.number(prior_row.get(key)) or np.nan for key in KEYS}
            anchor,details=self.tracker.fit(uid,reference,rows,prior,usable)
            if anchor:
                for frame_uid in usable:
                    cv2.setRNGSeed(0)
                    if frame_uid==str(reference.frame_uid):
                        # fit() already verified this exact image and rectangle.
                        zones[frame_uid]=anchor
                        evidence[frame_uid]=details
                        continue
                    mapped=register_neighbor(images[str(reference.frame_uid)],images[frame_uid],anchor)
                    if mapped:
                        visible=visible_transparent_zone(images[frame_uid],mapped,self.tracker.settings)
                        if visible['accepted']:
                            zones[frame_uid]=visible['local_zone']
                            evidence[frame_uid]={**details,**{k:v for k,v in visible.items()
                                                           if k not in ('accepted','local_zone')}}
            recovery_type='fox_temporal_alpha_fit'
            failure_reason=details.get('reason','fox_zone_unresolved')
        else:
            for frame_uid in usable:
                fit=refine_zone(images[frame_uid],frame_lookup[frame_uid])
                if not fit or not fit['accepted']:
                    game_prior=self.game_priors.get(uid.split('_')[0])
                    if game_prior:
                        fit=refine_zone(images[frame_uid],game_prior)
                if fit and fit['accepted']:
                    zones[frame_uid]={key:fit[key] for key in KEYS}
                    evidence[frame_uid]=dict(reason='recovered_four_edge_frame',zone_edge_score=fit['edge_score'],zone_edge_min_support=fit['edge_min_support'])
            if selected_uid in usable and selected_uid not in zones:
                transferred=[]
                for source_uid,source_zone in zones.items():
                    cv2.setRNGSeed(0)
                    mapped=register_neighbor(images[source_uid],images[selected_uid],source_zone)
                    if mapped:transferred.append(mapped)
                if len(transferred)>=3:
                    from refine_broadcast_zone import box_edges
                    edges=np.stack([box_edges(z) for z in transferred])
                    median=np.median(edges,axis=0)
                    kept=[z for z,e in zip(transferred,edges) if np.max(abs(e-median))<=3]
                    if len(kept)>=3:
                        zones[selected_uid]={key:float(np.median([z[key] for z in kept])) for key in KEYS}
                        evidence[selected_uid]=dict(reason='recovered_four_edge_frame',zone_edge_score=.85,zone_edge_min_support='',temporal_support=len(kept))
            recovery_type='recovered_four_edge_frame'
            failure_reason='no_supported_zone_in_usable_frames'
        for (_,row),view in zip(rows.iterrows(),view_results):
            frame_uid=str(row.frame_uid)
            self.frame_records.append(dict(frame_uid=frame_uid,pitch_uid=uid,
                zone_refinement_status=recovery_type if frame_uid in zones else 'recovery_rejected',
                frame_view_status=view['reason'],image_width=1280,image_height=720,
                **(zones.get(frame_uid) or {key:'' for key in KEYS})))
        if not zones:
            return self.reject(base,failure_reason)
        record=dict(base)
        if selected_uid in zones:
            chosen_uid=selected_uid
        else:
            pose_subset=self.poses[self.poses.pitch_uid.astype(str)==uid].reset_index()
            valid_rows=rows[rows.frame_uid.astype(str).isin(usable)]
            motion=glove_selector.delivery_motion(valid_rows,self.args,pose_subset)
            candidates=[]
            for frame_uid,zone in zones.items():
                row=frame_lookup[frame_uid]
                candidates.extend(glove_selector.candidate_rows(pd.DataFrame([row]),pd.Series(zone),self.args,motion))
            chosen=glove_selector.choose_candidate(candidates,self.args,motion)
            if chosen is None or chosen.get('track_frame_count',0)<3 or chosen.get('stable_run_count',0)<2:
                return self.reject(base,'no_stable_pre_pitch_glove_in_usable_view')
            chosen_uid=str(chosen['row'].frame_uid)
            zone_row=pd.Series({**base,**zones[chosen_uid]})
            record.update(glove_selector.output_record(uid,zone_row,chosen,len(candidates),sum(c['plausible'] for c in candidates),motion,self.args))
        record.update(zones[chosen_uid])
        detail=evidence[chosen_uid]
        record.update(zone_refinement_status=recovery_type,zone_source=recovery_type,
                      zone_anchor_source='fox_alpha_tracking' if broadcast['is_fox'] else 'roboflow_selected_frame',
                      zone_registration_source=recovery_type,zone_confidence=.85 if broadcast['is_fox'] else detail['zone_edge_score'],
                      recovery_reason=detail['reason'],recovery_original_frame_uid=base.get('frame_uid',''),
                      recovery_frame_changed='yes' if chosen_uid!=base.get('frame_uid') else 'no',
                      frame_view_status='usable_pitching_view',image_width=1280,image_height=720,
                      fox_logo_score=broadcast['score'] if broadcast['is_fox'] else '',
                      zone_refinement_frame_support=detail.get('fox_frame_support',1),
                      zone_refinement_reference_frame_uid=detail.get('fox_reference_frames',chosen_uid),
                      zone_edge_score=detail.get('zone_edge_score',''),zone_edge_min_support=detail.get('zone_edge_min_support',''))
        for key in ['fox_frame_support','fox_candidate_count','fox_temporal_error_px','fox_score','fox_reference_frames',
                    'fox_visible_fraction','fox_contrast_px','fox_temporal_visible_fraction']:
            record[key]=detail.get(key,'')
        tx,ty=recompute_target(pd.Series(record))
        record['vision_target_x_01']='' if tx is None else tx
        record['vision_target_y_01']='' if ty is None else ty
        record['status']='ok' if tx is not None else 'missing_required_detection'
        record['error']='' if tx is not None else 'missing_setup_glove'
        return record

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--dense',type=Path,required=True)
    parser.add_argument('--poses',type=Path,required=True)
    parser.add_argument('--videos',type=Path,default=ROOT/'.cache/zone_recovery_videos')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frame-output',type=Path,required=True)
    parser.add_argument('--audit-output',type=Path,required=True)
    args=parser.parse_args()
    with args.predictions.open(newline='') as handle:
        source=list(csv.DictReader(handle))
    pipeline=RecoveryPipeline(pd.read_csv(args.dense),pd.read_csv(args.poses),args.videos)
    protected=pd.DataFrame([r for r in source if r.get('zone_refinement_status') in PROTECTED])
    if not protected.empty:
        protected['game']=protected.pitch_uid.str.split('_').str[0]
        for game,rows in protected.groupby('game'):
            pipeline.game_priors[game]=rows[list(KEYS)].apply(pd.to_numeric,errors='coerce').median().to_dict()
    records=[]
    for row in source:
        record=pipeline.recover(row)
        records.append(record)
        if row.get('zone_refinement_status') in PROTECTED:
            assert all(record.get(key)==value for key,value in row.items()),'Protected row changed'
        else:
            print(f"{row['pitch_uid']}: {record['zone_refinement_status']} / {record.get('recovery_reason','')}",flush=True)
    write_records(args.output,records)
    write_records(args.frame_output,pipeline.frame_records,
                  ['frame_uid','pitch_uid','zone_refinement_status','frame_view_status','image_width','image_height',*KEYS])
    write_records(args.audit_output,pipeline.tracker.diagnostics,
                  ['pitch_uid','frame_uid',*KEYS,'fox_score','fox_min_support','fox_side_support','fox_corner_score','accepted'])
    print(pd.Series([row['zone_refinement_status'] for row in records]).value_counts().to_string(),flush=True)


if __name__=='__main__':main()
