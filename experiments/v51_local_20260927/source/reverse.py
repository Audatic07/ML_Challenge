"""Bounded reverse retrieval over ALL supplied S1 records, with no label lookup."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import gc,json,time
from pathlib import Path
import numpy as np
import polars as pl
from rapidfuzz import fuzz,process
from stage2.text_norm_v5 import normalise

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parents[1]/'student_resource/dataset/train'
KEEP=['country','core','concat','skel','addr_norm','addr_tok','house']
def collect(x):return x.collect(engine='streaming')
def norm(f):return normalise(f.with_columns(pl.col(pl.String).fill_null('')))
def keys(frame,idcol):
    views=[pl.col('core'),pl.col('core').str.split(' ').list.sort().list.join(' '),pl.col('concat'),
           pl.col('skel'),pl.col('addr_norm'),pl.concat_str([pl.col('house').fill_null(''),pl.col('addr_tok')],separator=' ')]
    out=[]
    for i,expr in enumerate(views):
        part=frame.select(idcol,'country',expr.alias('_key')).filter(pl.col('_key').str.len_chars()>=5)
        part=part.select(idcol,pl.concat_str([pl.col('country'),pl.lit(str(i)),pl.col('_key')],separator='|').hash(seed=20260927).alias('key'))
        out.append(part)
    return pl.concat(out).unique()

def main():
    started=time.monotonic()
    idx=ROOT/'reverse_index';idx.mkdir(exist_ok=True)
    src=pl.scan_csv(DATA/'train_source1.tsv',separator='\t',quote_char=None,infer_schema=False).with_row_index('s1')
    total=2206821
    for start in range(0,total,100000):
        dest=idx/f'norm_{start:07d}.parquet'
        if dest.exists():continue
        f=norm(collect(src.slice(start,100000))).select('s1',*KEEP)
        keys(f,'s1').write_parquet(idx/f'keys_{start:07d}.parquet')
        f.write_parquet(dest)
        if start%500000==0:print('S1 reverse index',start+len(f),'seconds',round(time.monotonic()-started),flush=True)
        del f;gc.collect()
    keyscan=pl.scan_parquet(str(idx/'keys_*.parquet'))
    # Suppress expensive common keys. This is an additional measured retrieval
    # feature, never a veto and never a replacement for the original candidates.
    valid=collect(keyscan.group_by('key').len().filter(pl.col('len')<=20).select('key'))
    index=collect(keyscan.join(valid.lazy(),on='key',how='semi'))
    print('Reverse keys retained',len(index),flush=True)
    target=pl.scan_parquet(str(ROOT/'target_text_*.parquet'))
    target_total=collect(target.select(pl.len())).item()
    cache=ROOT/'reverse_competitors';cache.mkdir(exist_ok=True)
    for start in range(0,target_total,20000):
        dest=cache/f'{start:07d}.parquet'
        if dest.exists():continue
        ts=norm(collect(target.slice(start,20000))).select(pl.col('entity_id').alias('target_id'),*KEEP)
        matches=keys(ts,'target_id').join(index,on='key').select('target_id','s1').unique()
        if len(matches):
            q=collect(pl.scan_parquet(str(idx/'norm_*.parquet')).join(matches.lazy().select('s1').unique(),on='s1',how='semi'))
            joined=matches.join(ts,on='target_id').join(q,on='s1',suffix='_q')
            out={}
            for key,col,scorer in [('n','core',fuzz.token_set_ratio),('nc','concat',fuzz.ratio),('a','addr_norm',fuzz.token_set_ratio),('ac','addr_norm',fuzz.ratio),('st','addr_tok',fuzz.token_set_ratio)]:
                a=joined[col].fill_null('');b=joined[col+'_q'].fill_null('')
                vals=process.cpdist(a.to_list(),b.to_list(),scorer=scorer,workers=2,dtype=np.float32)/100
                vals[((a=='')|(b=='')).to_numpy()]=0
                out[key]=vals
            out['joint']=np.sqrt(out['nc']*out['ac'])
            out['joint_set']=np.sqrt(out['n']*out['a'])
            result=joined.select('target_id','s1').with_columns([pl.Series(k,v) for k,v in out.items()])
            # Keep alternatives for each similarity criterion, not just one rank.
            ranked=pl.concat([result.sort(['target_id',k,'s1'],descending=[False,True,False]).group_by('target_id',maintain_order=True).head(3) for k in ['joint','joint_set','n','a']]).unique(subset=['target_id','s1'])
        else:
            ranked=pl.DataFrame(schema={'target_id':pl.String,'s1':pl.UInt32,**{k:pl.Float32 for k in ['n','nc','a','ac','st','joint','joint_set']}})
        ranked.write_parquet(dest)
        if start%100000==0:print('Reverse target search',start+len(ts),'/',target_total,'seconds',round(time.monotonic()-started),flush=True)
        del ts,matches;gc.collect()
    del index,valid;gc.collect()
    comp=pl.scan_parquet(str(cache/'*.parquet')).rename({'s1':'other_s1'})
    outdir=ROOT/'features_reverse';outdir.mkdir(exist_ok=True)
    for path in sorted((ROOT/'features').glob('*.parquet')):
        dest=outdir/path.name
        if dest.exists():continue
        frame=pl.read_parquet(path)
        other=collect(frame.lazy().select('s1','target_id').join(comp,on='target_id').filter(pl.col('s1')!=pl.col('other_s1')).group_by('s1','target_id').agg(*[pl.col(k).max().alias('full_reverse_'+k) for k in ['n','nc','a','ac','st','joint','joint_set']],pl.len().alias('full_reverse_count')))
        frame=frame.join(other,on=['s1','target_id'],how='left').with_columns(pl.col('^full_reverse_.*$').fill_null(0).cast(pl.Float32))
        frame=frame.with_columns((pl.col('q_cc_ratio')/100-pl.col('full_reverse_nc')).alias('full_reverse_name_gap'),(pl.col('q_a_ratio')/100-pl.col('full_reverse_ac')).alias('full_reverse_address_gap'),((pl.col('q_cc_ratio').clip(lower_bound=0)*pl.col('q_a_ratio').clip(lower_bound=0)).sqrt()/100-pl.col('full_reverse_joint')).alias('full_reverse_joint_gap'))
        frame.write_parquet(dest)
    (ROOT/'reverse_manifest.json').write_text(json.dumps({'method':'label-free exact normalized name/address/skeleton keys over complete S1 population; max 20 postings per key; top alternatives by multiple similarities','population_rows':total,'target_rows':target_total,'seconds':time.monotonic()-started,'external_data':False,'labels_used_as_features':False},indent=2))
    print('REVERSE COMPLETE',round(time.monotonic()-started),flush=True)

if __name__=='__main__':main()
