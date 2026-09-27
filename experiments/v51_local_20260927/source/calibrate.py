import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import argparse,json,time
from pathlib import Path
import numpy as np
import polars as pl
from selection import threshold,expected,per_query

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--models',nargs='+',default=['lightgbm','xgboost']);args=ap.parse_args()
    qs=pl.read_parquet(ROOT/'queries.parquet').filter(pl.col('partition')==2)
    truth=pl.read_parquet(ROOT/'truth.parquet').join(qs.select('s1'),on='s1')
    frames={name:pl.read_parquet(ROOT/f'{name}_calibrate.parquet').sort('s1','t') for name in args.models}
    base=frames[args.models[0]]
    for other in frames.values():assert base.select('s1','t').equals(other.select('s1','t'))
    def score(pred):return float(per_query(qs,truth,pred)['f'].mean())
    oracle=score(base.select('s1','target_id').join(truth,on=['s1','target_id']))
    candidates={'baseline':base.with_columns(pl.col('p1').alias('score'))}
    candidates.update(frames)
    for name,frame in frames.items():
        for weight in [.25,.5,.75]:
            candidates[f'{name}_rawblend_{weight}']=frame.with_columns((weight*pl.col('score')+(1-weight)*pl.col('p1')).alias('score'))
    if len(frames)>1:
        import itertools
        for left,right in itertools.combinations(args.models,2):
            candidates[f'ensemble:{left}:{right}']=base.with_columns(pl.Series('score',(frames[left]['score'].to_numpy()+frames[right]['score'].to_numpy())/2))
    results=[]
    for name,frame in candidates.items():
        grid=[]
        for cut in np.arange(.45,.951,.01):
            grid.append((score(threshold(frame,float(cut))),float(cut)))
        best,cut=max(grid)
        results.append({'name':name,'policy':'threshold','threshold':cut,'macro_f05':best})
        print(name,'threshold',round(cut,4),'F0.5',best,flush=True)
    # Set optimization is evaluated only for the two best calibrated model choices.
    best_names=[r['name'] for r in sorted(results,key=lambda r:r['macro_f05'],reverse=True)[:2]]
    for name in best_names:
        frame=candidates[name]
        for temp,bias in [(1.,0.),(1.,-.3),(1.,-.6),(1.,-.9),(1.,.3),(.8,0.),(.8,-.3),(1.2,0.),(1.2,-.3)]:
            value=score(expected(frame,bias=bias,temperature=temp))
            results.append({'name':name,'policy':'expected_f05','bias':bias,'temperature':temp,'macro_f05':value})
            print(name,'expected-F',temp,bias,value,flush=True)
    champion=max(results,key=lambda r:r['macro_f05'])
    report={'partition':'10000 second-stage calibration queries from original v5.1 tuning set','models_considered':args.models,
            'survivor_oracle_u':oracle,'results':results,'champion':champion,
            'original_threshold_baseline':score(threshold(candidates['baseline'],.785)),
            'check_evaluated':False,'audit_evaluated':False}
    (ROOT/'calibration.json').write_text(json.dumps(report,indent=2))
    print('CHAMPION',champion,'ORACLE',oracle,flush=True)

if __name__=='__main__':main()
