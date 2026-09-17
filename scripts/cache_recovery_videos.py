"""Cache only explicitly queued public MLB source clips for zone recovery."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse
import pandas as pd
import requests
import cv2

from detect_fox_zone import FoxBroadcastClassifier

ROOT=Path(__file__).resolve().parents[1]
PROTECTED={'visible_four_edge_fit','registered_temporal_fit','manual_frame_override'}


def discovery_rows(predictions,matches):
    predictions=pd.read_csv(predictions,keep_default_na=False)
    matches=pd.read_csv(matches,keep_default_na=False)
    urls=matches.drop_duplicates('pitch_uid').set_index('pitch_uid')['direct_mp4_url'].to_dict()
    classifier=FoxBroadcastClassifier(ROOT/'models/fox_zone_logos')
    rows=[]
    for _,row in predictions.iterrows():
        if row.get('zone_refinement_status') in PROTECTED:continue
        image=cv2.imread(str(ROOT/str(row.get('image_path',''))))
        if not classifier.classify(image)['is_fox']:continue
        url=urls.get(row.pitch_uid) or row.get('direct_mp4_url','')
        if url:rows.append(dict(pitch_uid=row.pitch_uid,direct_mp4_url=url))
    return pd.DataFrame(rows,columns=['pitch_uid','direct_mp4_url'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--queue',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--matches',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    if args.queue:
        rows=pd.read_csv(args.queue).drop_duplicates('pitch_uid')
    elif args.predictions and args.matches:
        rows=discovery_rows(args.predictions,args.matches).drop_duplicates('pitch_uid')
    else:
        parser.error('Use --queue or both --predictions and --matches')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if rows.empty:
        print('No unresolved FOX clips require caching.',flush=True)
        return
    def download(row):
        uid=str(row.pitch_uid)
        if not all(c.isdigit() or c=='_' for c in uid):raise ValueError('Invalid pitch identifier')
        url=row.direct_mp4_url
        if urlparse(url).scheme!='https' or urlparse(url).hostname!='sporty-clips.mlb.com':
            raise ValueError('Only existing public MLB clip URLs are allowed')
        path=args.output_dir/f'{uid}.mp4'
        if path.exists():return f'{uid}: cached'
        temporary=path.with_suffix('.mp4.tmp')
        try:
            with requests.get(url,timeout=(10,60),stream=True) as response:
                response.raise_for_status()
                total=0
                with temporary.open('wb') as handle:
                    for chunk in response.iter_content(1024*1024):
                        total+=len(chunk)
                        if total>100*1024*1024:raise ValueError('Clip exceeds 100 MB limit')
                        handle.write(chunk)
            temporary.replace(path)
            return f'{uid}: downloaded {total} bytes'
        finally:
            if temporary.exists():temporary.unlink()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(download,(row for _,row in rows.iterrows())):
            print(result,flush=True)


if __name__=='__main__':main()
