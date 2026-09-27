import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','3')
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parent
def collect(x):return x.collect(engine='streaming')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('model',choices=['lightgbm','xgboost']);ap.add_argument('--tag');ap.add_argument('--features',default='features');args=ap.parse_args()
    tag=args.tag or args.model
    t0=time.monotonic()
    qs=pl.read_parquet(ROOT/'queries.parquet')
    truth=pl.read_parquet(ROOT/'truth.parquet')
    source=pl.scan_parquet(str(ROOT/args.features/'*.parquet'))
    names=[c for c in source.collect_schema().names() if c not in ['s1','t','target_id']]
    label=truth.lazy().with_columns(pl.lit(1,dtype=pl.UInt8).alias('y'))
    held=truth.join(qs.filter(pl.col('partition')!=0).select('s1'),on='s1').select('target_id').unique()
    def load(part,exclude=False):
        f=source.join(qs.filter(pl.col('partition')==part).lazy().select('s1'),on='s1',how='semi')
        if exclude:f=f.join(held.lazy(),on='target_id',how='anti')
        frame=collect(f.join(label,on=['s1','target_id'],how='left').with_columns(pl.col('y').fill_null(0)))
        keys=frame.select('s1','t','target_id','y','p1')
        x=frame.select(names).to_numpy(order='c');y=frame['y'].to_numpy()
        return keys,x,y
    train_keys,x,y=load(0,True)
    stop_keys,xs,ys=load(1)
    print(args.model,'fit',x.shape,'stop',xs.shape,'seconds',round(time.monotonic()-t0),flush=True)
    if args.model=='lightgbm':
        import lightgbm as lgb
        params=dict(objective='binary',metric='binary_logloss',learning_rate=.04,num_leaves=63,min_data_in_leaf=120,
                    feature_fraction=.9,bagging_fraction=.85,bagging_freq=1,lambda_l2=5,seed=27092026,
                    num_threads=3,deterministic=True,force_col_wise=True,verbosity=-1)
        d=lgb.Dataset(x,label=y,feature_name=names,free_raw_data=True)
        ds=lgb.Dataset(xs,label=ys,reference=d,free_raw_data=True)
        d.construct();ds.construct();del x,xs;gc.collect()
        model=lgb.train(params,d,num_boost_round=1800,valid_sets=[ds],callbacks=[lgb.early_stopping(100),lgb.log_evaluation(100)])
        model.save_model(str(ROOT/f'{tag}.txt'))
        best=model.best_iteration
        del d,ds;gc.collect()
        predict=lambda x:model.predict(x,num_threads=3)
    else:
        import xgboost as xgb
        params=dict(objective='binary:logistic',eval_metric='logloss',learning_rate=.045,max_depth=6,min_child_weight=20,
                    subsample=.85,colsample_bytree=.9,reg_lambda=8,tree_method='hist',nthread=3,seed=27092026)
        d=xgb.QuantileDMatrix(x,label=y,feature_names=names,max_bin=128)
        ds=xgb.QuantileDMatrix(xs,label=ys,feature_names=names,ref=d,max_bin=128)
        del x,xs;gc.collect();params['max_bin']=128
        model=xgb.train(params,d,num_boost_round=1800,evals=[(ds,'stop')],early_stopping_rounds=100,verbose_eval=100)
        best=model.best_iteration
        model.save_model(ROOT/f'{tag}.ubj')
        del d,ds;gc.collect()
        predict=lambda x:model.inplace_predict(x,iteration_range=(0,best+1))
    del train_keys,stop_keys,y,ys;gc.collect()
    # Never evaluate or predict the final check partition during model selection.
    keys,xc,yc=load(2)
    keys.with_columns(pl.Series('score',predict(xc).astype(np.float32))).write_parquet(ROOT/f'{tag}_calibrate.parquet')
    (ROOT/f'{tag}_manifest.json').write_text(json.dumps({'model':args.model,'tag':tag,'feature_dir':args.features,'features':names,'params':params,
        'best_iteration':best,'seconds':time.monotonic()-t0,'excluded_heldout_truth_targets':len(held),
        'fit_queries':65000,'stop_queries':10000,'calibrate_queries':10000,'check_evaluated':False,
        'audit_evaluated':False},indent=2))
    print('TRAIN COMPLETE',args.model,'best',best,'seconds',round(time.monotonic()-t0),flush=True)

if __name__=='__main__':main()
