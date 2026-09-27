import itertools
import numpy as np
import polars as pl
from selection import expected_values,expected,per_query

def test_expected_matches_enumeration():
    p=np.array([[.8,.4,.1],[1.,0.,0.],[0.,0.,0.]])
    actual=expected_values(p)
    brute=np.zeros_like(actual)
    for i,row in enumerate(p):
        for y in itertools.product([0,1],repeat=3):
            mass=np.prod([row[j] if y[j] else 1-row[j] for j in range(3)])
            for size in range(4):
                f=float(size==0) if not sum(y) else 5*sum(y[:size])/(4*size+sum(y))
                brute[i,size]+=mass*f
    np.testing.assert_allclose(actual,brute,atol=1e-12)

def test_zero_and_single_survivor():
    empty=pl.DataFrame(schema={'s1':pl.UInt32,'target_id':pl.String,'score':pl.Float32})
    assert expected(empty).height==0
    one=pl.DataFrame({'s1':[0],'target_id':['x'],'score':[.9]})
    assert expected(one).height==1
    q=pl.DataFrame({'s1':[0,1,2],'country':['x']*3})
    truth=pl.DataFrame({'s1':[0,2],'target_id':['x','z']})
    assert per_query(q,truth,expected(one))['f'].to_list()==[1.,1.,0.]

def test_stage2_one_and_two_candidates():
    from stage2.base import features,TEXT
    for n in [1,2]:
        text=pl.DataFrame({**{c:['same']*n for c in TEXT},'nlen':[4]*n,'alen':[4]*n,'src':[2]*n})
        s=pl.DataFrame({'s1':[0]*n,'t':list(range(n)),'target_id':[f't{i}' for i in range(n)],'p':[.9,.2][:n],'rank':list(range(n)),'qi':[0]*n,'ti':list(range(n))})
        out=features(s,text.head(1),text,1)
        assert len(out)==n
        assert out['p_3'].to_list()==[0.]*n

def test_macro_counts_misses_false_positives_and_singletons():
    q=pl.DataFrame({'s1':[0,1,2,3],'country':['x']*4})
    t=pl.DataFrame({'s1':[0,0,0,1,1,1],'target_id':['a','b','c','d','e','f']})
    p=pl.DataFrame({'s1':[0,0,1,1,1,1,3],'target_id':['a','b','d','e','f','wrong','also_wrong']})
    f=per_query(q,t,p).sort('s1')['f'].to_numpy()
    np.testing.assert_allclose(f,[10/11,15/19,1,0])
