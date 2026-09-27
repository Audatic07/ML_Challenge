import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','3')
import hashlib,json,time,gc
from pathlib import Path
import numpy as np
import polars as pl
from selection import expected,threshold,per_query

ROOT=Path(__file__).resolve().parent
INPUT=ROOT.parent/'v51_received_20260927'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if (ROOT/'final_check.json').exists():raise RuntimeError('Reserved check already evaluated; do not reuse it for tuning.')
    calibration=json.loads((ROOT/'calibration.json').read_text())
    champion=calibration['champion']
    frozen={'champion':champion,'calibration_sha256':sha(ROOT/'calibration.json'),
            'queries_sha256':sha(ROOT/'queries.parquet'),'candidate_policy':{'top_k':16,'floor':.0001},
            'selection_code_sha256':sha(ROOT/'selection.py'),'models':{}}
    for manifest in ROOT.glob('*_manifest.json'):
        info=json.loads(manifest.read_text())
        if 'model' in info:
            name=manifest.name.removesuffix('_manifest.json')
            filename=name+('.txt' if info['model']=='lightgbm' else '.ubj')
            frozen['models'][name]={'file':filename,'sha256':sha(ROOT/filename)}
    (ROOT/'champion_frozen.json').write_text(json.dumps(frozen,indent=2))
    qs=pl.read_parquet(ROOT/'queries.parquet').filter(pl.col('partition')==3)
    truth=pl.read_parquet(ROOT/'truth.parquet').join(qs.select('s1'),on='s1')
    cache={}
    def scores(name):
        if name in cache:return cache[name]
        if name.startswith('ensemble:'):
            _,left,right=name.split(':')
            a=scores(left);b=scores(right)
            assert a.select('s1','t').equals(b.select('s1','t'))
            result=a.with_columns(pl.Series('score',(a['score'].to_numpy()+b['score'].to_numpy())/2))
        elif '_rawblend_' in name:
            model,weight=name.rsplit('_rawblend_',1);weight=float(weight)
            result=scores(model).with_columns((weight*pl.col('score')+(1-weight)*pl.col('p1')).alias('score'))
        elif name=='baseline':
            result=pl.read_parquet(ROOT/'survivors.parquet').join(qs.select('s1'),on='s1').rename({'p':'score'}).sort('s1','t')
        else:
            meta=json.loads((ROOT/f'{name}_manifest.json').read_text())
            frame=(pl.scan_parquet(str(ROOT/meta.get('feature_dir','features')/'*.parquet')).join(qs.lazy().select('s1'),on='s1',how='semi').sort('s1','t').collect(engine='streaming'))
            x=frame.select(meta['features']).to_numpy(order='c')
            if meta['model']=='lightgbm':
                import lightgbm as lgb
                model=lgb.Booster(model_file=str(ROOT/f'{name}.txt'))
                p=model.predict(x,num_threads=3)
            else:
                import xgboost as xgb
                model=xgb.Booster(model_file=str(ROOT/f'{name}.ubj'));model.set_param({'nthread':3})
                p=model.inplace_predict(x,iteration_range=(0,meta['best_iteration']+1))
            result=frame.select('s1','t','target_id','p1').with_columns(pl.Series('score',p.astype(np.float32)))
            del frame,x,model;gc.collect()
        cache[name]=result
        return result
    scored=scores(champion['name'])
    pred=(threshold(scored,champion['threshold']) if champion['policy']=='threshold' else expected(scored,bias=champion['bias'],temperature=champion['temperature']))
    actual=per_query(qs,truth,pred).sort('s1')
    base_scores=(pl.scan_parquet(INPUT/'model/tune_scores.parquet').filter(pl.col('p')>=.785).join(qs.lazy().select('s1'),on='s1',how='semi').rename({'p':'score'}).collect(engine='streaming'))
    baseline=per_query(qs,truth,threshold(base_scores,.785)).sort('s1')
    old=pl.read_parquet(INPUT/'baseline_reproduced_per_query.parquet').join(qs.select('s1'),on='s1').sort('s1')
    oracle=per_query(qs,truth,scored.select('s1','target_id').join(truth,on=['s1','target_id']))
    new_f=actual['f'].to_numpy();old_f=baseline['f'].to_numpy()
    rng=np.random.default_rng(27092026);means=[];deltas=[]
    for _ in range(1000):
        idx=rng.integers(0,len(qs),len(qs));means.append(float(new_f[idx].mean()));deltas.append(float((new_f-old_f)[idx].mean()))
    report={'evaluation':'15000 reserved second-stage development queries; these were in the original v5.1 tuning set, so this is not a new competition audit',
        'queries':len(qs),'champion':champion,'baseline_same_queries_macro_f05':float(old_f.mean()),
        'published_v51_assignment_on_same_queries':old['reproduced_f'].mean(),
        'new_macro_f05':float(new_f.mean()),'paired_gain':float((new_f-old_f).mean()),
        'candidate_oracle_u':float(oracle['f'].mean()),'matching_gap':float(oracle['f'].mean()-new_f.mean()),
        'score_bootstrap_95_interval':np.quantile(means,[.025,.975]).tolist(),
        'gain_bootstrap_95_interval':np.quantile(deltas,[.025,.975]).tolist(),
        'score_one_sided_95_lower':float(np.quantile(means,.05)),
        'by_country':actual.group_by('country').agg(pl.len().alias('queries'),pl.col('f').mean().alias('macro_f05')).to_dicts(),
        'singleton_queries':actual.filter(pl.col('g')==0).height,
        'singleton_f05':actual.filter(pl.col('g')==0)['f'].mean(),
        'zero_candidate_queries':len(qs)-scored['s1'].n_unique(),
        'candidate_pairs':len(scored),'predicted_pairs':len(pred),
        'external_data_used':False,'competition_audit_evaluated':False,
        'target_098_met':bool(new_f.mean()>=.98),'target_099_met':bool(new_f.mean()>=.99),
        'no_more_tuning_after_this_check':True}
    actual.with_columns(baseline['f'].alias('baseline_f')).write_parquet(ROOT/'final_check_per_query.parquet')
    pred.write_parquet(ROOT/'final_check_predictions.parquet')
    scored.write_parquet(ROOT/'final_check_scores.parquet')
    (ROOT/'final_check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
