import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import gc, hashlib, json, time
from pathlib import Path
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein
from stage2 import base as b

ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[1]
INPUT=ROOT.parent/'v51_received_20260927'
DATA=WORK/'student_resource/dataset/train'
def collect(q):return q.collect(engine='streaming')
def readraw(path):return pl.scan_csv(path,separator='\t',quote_char=None,infer_schema=False).with_columns(pl.col(pl.String).fill_null(''))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def extra(a,c):
    out={}
    for col,short in [('core','name'),('addr_norm','addr'),('addr_tok','street')]:
        av=a[col].fill_null('');cv=c[col].fill_null('')
        left,right=av.to_list(),cv.to_list()
        missing=((av=='')|(cv=='')).to_numpy()
        for tag,scorer in [('ratio',fuzz.ratio),('partial',fuzz.partial_ratio),('sort',fuzz.token_sort_ratio),('lev',Levenshtein.normalized_similarity)]:
            x=process.cpdist(left,right,scorer=scorer,workers=2,dtype=np.float32)
            x[missing]=-1;out[f'{short}_{tag}']=x
        frame=pl.DataFrame({'a':av,'b':cv}).with_columns(pl.col('a','b').str.split(' ').list.eval(pl.element().filter(pl.element()!='')).list.unique())
        inter=frame.select(pl.col('a').list.set_intersection('b').list.len()).to_series().to_numpy().astype(np.float32)
        la=frame['a'].list.len().to_numpy();lb=frame['b'].list.len().to_numpy()
        out[f'{short}_precision']=inter/np.maximum(1,lb)
        out[f'{short}_recall']=inter/np.maximum(1,la)
        out[f'{short}_jaccard']=inter/np.maximum(1,la+lb-inter)
        out[f'{short}_length_ratio']=np.minimum(av.str.len_chars().to_numpy(),cv.str.len_chars().to_numpy())/np.maximum(1,np.maximum(av.str.len_chars().to_numpy(),cv.str.len_chars().to_numpy()))
    nums=pl.DataFrame({'a':a['addr_norm'],'b':c['addr_norm']}).with_columns(pl.col('a','b').fill_null('').str.extract_all(r'\d+').list.eval(pl.element().str.strip_chars_start('0')).list.unique())
    ni=nums.select(pl.col('a').list.set_intersection('b').list.len()).to_series().to_numpy().astype(np.float32)
    na=nums['a'].list.len().to_numpy();nb=nums['b'].list.len().to_numpy()
    out.update(number_shared=ni,number_query=na,number_target=nb,number_missing=na-ni,number_extra=nb-ni,number_jaccard=ni/np.maximum(1,na+nb-ni))
    return out

def main():
    started=time.monotonic()
    plan=json.loads((INPUT/'plan.json').read_text())
    session=json.loads((ROOT/'session.json').read_text())
    queries=pl.DataFrame({'s1':pl.Series(plan['tune_rows'],dtype=pl.UInt32),'country':plan['tune_country']})
    perm=np.random.default_rng(session['partition_seed']).permutation(len(queries))
    parts=np.empty(len(queries),dtype=np.uint8)
    for part,inds in enumerate(np.split(perm,[65000,75000,85000])):parts[inds]=part
    queries=queries.with_columns(pl.Series('partition',parts))
    queries.write_parquet(ROOT/'queries.parquet')
    qraw=collect(readraw(DATA/'train_source1.tsv').with_row_index('s1').join(queries.lazy().select('s1'),on='s1',how='semi'))
    qraw.write_parquet(ROOT/'query_text.parquet')
    labels=collect(readraw(DATA/'train_ground_truth.tsv').join(qraw.lazy().select('s1',pl.col('entity_id').alias('source1_entity_id')),on='source1_entity_id'))
    assert labels['s1'].n_unique()==100000
    truth=(labels.with_columns(pl.col('matched_entity_ids').str.split(',')).explode('matched_entity_ids').filter(pl.col('matched_entity_ids')!='').select('s1',pl.col('matched_entity_ids').alias('target_id')))
    assert not truth.is_duplicated().any()
    assert truth['target_id'].n_unique()==len(truth), 'Shared truth targets require group-aware partitions'
    truth.write_parquet(ROOT/'truth.parquet')
    del labels
    scorepath=ROOT/'survivors.parquet'
    if not scorepath.exists():
        small=collect(pl.scan_parquet(INPUT/'model/tune_scores.parquet').filter(pl.col('p')>=0.0001))
        surv=b.survivors(small,16,0.0001)
        surv.write_parquet(scorepath)
        del small,surv;gc.collect()
    surv=pl.read_parquet(scorepath)
    print('Survivors',len(surv),'queries',surv['s1'].n_unique(),flush=True)
    targetids=surv.select(pl.col('target_id').alias('entity_id')).unique()
    for source in [2,3]:
        dest=ROOT/f'target_text_{source}.parquet'
        if not dest.exists():
            selected=collect(readraw(DATA/f'train_source{source}.tsv').join(targetids.lazy(),on='entity_id',how='semi').with_columns(pl.lit(source,dtype=pl.UInt8).alias('src')))
            selected.write_parquet(dest);print('Cached source',source,len(selected),flush=True)
            del selected;gc.collect()
    # Reverse competition comes from predictions only, never truth labels.
    reverse=(surv.group_by('target_id').agg(pl.col('p').max().alias('rev_max'),pl.col('p').sum().alias('rev_sum'),pl.len().alias('rev_count'),pl.col('p').sort(descending=True).implode().list.get(1,null_on_oob=True).fill_null(0).alias('rev_second')))
    surv=surv.join(reverse,on='target_id',how='left').with_columns((pl.col('rev_max')-pl.col('p')).alias('rev_gap'),(pl.col('rev_sum')-pl.col('p')).alias('rev_others'))
    feature_dir=ROOT/'features';feature_dir.mkdir(exist_ok=True)
    for start in range(0,len(queries),2000):
        dest=feature_dir/f'{start:06d}.parquet'
        if dest.exists():continue
        rows=queries['s1'].slice(start,2000)
        ss=surv.filter(pl.col('s1').is_in(rows.implode())).sort('s1','rank')
        if not len(ss):continue
        q=qraw.filter(pl.col('s1').is_in(rows.implode())).with_row_index('qi')
        targets=ss.select(pl.col('target_id').alias('entity_id')).unique()
        t=collect(pl.scan_parquet(str(ROOT/'target_text_*.parquet')).join(targets.lazy(),on='entity_id',how='semi')).with_row_index('ti')
        ss=ss.join(q.select('s1','qi'),on='s1',how='left').join(t.select(pl.col('entity_id').alias('target_id'),'ti'),on='target_id',how='left').sort('s1','rank')
        assert ss['ti'].null_count()==0
        qn=b.text_views(q)
        tn=b.text_views(t).with_columns(t['src'])
        feats=b.features(ss,qn,tn,2)
        extras=extra(qn.gather(ss['qi']),tn.gather(ss['ti']))
        for col in ['rev_max','rev_sum','rev_count','rev_second','rev_gap','rev_others']:extras[col]=ss[col].to_numpy()
        # Explicit per-anchor similarities preserve direction and disagreement,
        # beyond the published stage-two maximum-similarity summaries.
        for rank in range(4):
            anchors=ss.filter(pl.col('rank')==rank).select('s1',pl.col('ti').alias('anchor_ti'),pl.col('p').alias('anchor_p'))
            aligned=ss.select('s1','ti').join(anchors,on='s1',how='left')
            missing=(aligned['anchor_ti'].is_null()|(aligned['anchor_ti']==aligned['ti'])).to_numpy()
            anc=tn.gather(aligned['anchor_ti'].fill_null(0))
            block=b.pair_block(tn.gather(ss['ti']),anc,2,'')
            for key in ['n_tset','n_tsort','cc_ratio','cc_jw','a_tset','a_ratio','at_tset','hs_eq','pc_eq','n_eq','a_eq']:
                vals=block[key];vals[missing]=-1;extras[f'anchor{rank}_{key}']=vals
            vals=aligned['anchor_p'].fill_null(0).to_numpy().copy();vals[missing]=0
            extras[f'anchor{rank}_p']=vals
        feats=feats.with_columns([pl.Series(k,np.asarray(v,dtype=np.float32)) for k,v in extras.items()])
        feats.write_parquet(dest)
        if start%10000==0:print('Features',start+len(rows),'/',len(queries),'columns',len(feats.columns),'seconds',round(time.monotonic()-started),flush=True)
        del ss,q,t,qn,tn,feats,extras;gc.collect()
    (ROOT/'preparation.json').write_text(json.dumps({'seconds':time.monotonic()-started,'queries':len(queries),'survivors':len(surv),'candidate_policy':session['candidate_policy'],'source_model_sha':json.loads((INPUT/'model/model_manifest.json').read_text())['model_sha256'],'code_checksums':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),*list((ROOT/'stage2').glob('*.py'))]}},indent=2))
    print('PREPARATION COMPLETE',flush=True)

if __name__=='__main__':main()
