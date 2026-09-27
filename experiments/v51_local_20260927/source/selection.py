"""Set selection and exact per-query F0.5; empty queries always remain in evaluation."""
import numpy as np
import polars as pl

def per_query(queries,truth,pred):
    assert not pred.select('s1','target_id').is_duplicated().any()
    g=truth.group_by('s1').len().rename({'len':'g'})
    p=pred.group_by('s1').len().rename({'len':'n_pred'})
    tp=pred.join(truth,on=['s1','target_id']).group_by('s1').len().rename({'len':'tp'})
    out=queries.select('s1','country').join(g,on='s1',how='left').join(p,on='s1',how='left').join(tp,on='s1',how='left').fill_null(0)
    return out.with_columns(pl.when(pl.col('g')==0).then((pl.col('n_pred')==0).cast(pl.Float64)).otherwise(5*pl.col('tp')/(4*pl.col('n_pred')+pl.col('g'))).alias('f'))

def unique(scores):
    return scores.sort(['target_id','score','s1'],descending=[False,True,False]).unique(subset=['target_id'],keep='first',maintain_order=True)

def threshold(scores,cut):return unique(scores.filter(pl.col('score')>=cut)).select('s1','target_id')

def expected_values(prob):
    """Exact expected F0.5 for top-k sets under independent Bernoulli labels.

    prob is N x K, sorted descending and zero-padded. The independence and
    zero-positive-mass-outside-survivors assumptions are approximations to reality.
    """
    prob=np.asarray(prob,dtype=np.float64)
    n,k=prob.shape
    pre=np.zeros((n,k+1,k+1));suf=np.zeros_like(pre)
    pre[:,0,0]=1;suf[:,k,0]=1
    for j in range(k):
        p=prob[:,j,None]
        pre[:,j+1,:]=(1-p)*pre[:,j,:]
        pre[:,j+1,1:]+=p*pre[:,j,:-1]
    for j in range(k-1,-1,-1):
        p=prob[:,j,None]
        suf[:,j,:]=(1-p)*suf[:,j+1,:]
        suf[:,j,1:]+=p*suf[:,j+1,:-1]
    values=np.zeros((n,k+1));values[:,0]=pre[:,k,0]
    for size in range(1,k+1):
        for a in range(1,size+1):
            for c in range(k-size+1):
                values[:,size]+=pre[:,size,a]*suf[:,size,c]*(5*a/(4*size+a+c))
    return values

def expected(scores,bias=0.0,temperature=1.0):
    scores=unique(scores).sort(['s1','score','target_id'],descending=[False,True,False])
    if scores.is_empty():return scores.select('s1','target_id')
    scores=scores.with_columns(pl.int_range(pl.len()).over('s1').alias('_rank'))
    keys=scores['s1'].to_numpy();rows,inv=np.unique(keys,return_inverse=True)
    ranks=scores['_rank'].to_numpy();width=int(ranks.max())+1
    p=np.clip(scores['score'].to_numpy().astype(float),1e-7,1-1e-7)
    p=1/(1+np.exp(-(np.log(p/(1-p))+bias)/temperature))
    dense=np.zeros((len(rows),width));dense[inv,ranks]=p
    count=expected_values(dense).argmax(axis=1)
    return scores.filter(pl.Series(ranks<count[inv])).select('s1','target_id')
