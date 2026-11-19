from pathlib import Path
import csv, json, hashlib, collections, math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
ROOT=Path(__file__).resolve().parent
def read(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f,delimiter=';'))
def val(a,k): return float(a[k])
def close(a,b,atol=1e-6):
    np.testing.assert_allclose(a,b,atol=atol,rtol=1e-6)
def main():
    a=read('data/merged_geodata/regions_2021_attributes.csv')
    o=read('results/weights_order.csv')
    e=read('results/weights_edges.csv')
    p=read('results/regional_predictions.csv')
    saved={r['model']:r for r in read('results/model_comparison.csv')}
    coeff={(r['model'],r['parameter']):float(r['coefficient']) for r in read('results/model_coefficients.csv')}
    d={r['object_okato']:r for r in a}; pr={r['object_okato']:r for r in p}
    ids=[r['object_okato'] for r in sorted(o,key=lambda r:int(r['row_index']))]
    assert len(a)==len(o)==len(p)==len(d)==len(pr)==len(set(ids))==85
    assert set(ids)==set(d)==set(pr) and ids==sorted(ids)
    assert all(r['year']=='2021' for r in a)
    assert len({r['object_oktmo'] for r in a})==84
    x=np.array([val(d[i],'doctors_per_10000') for i in ids])
    y=np.array([val(d[i],'life_expectancy_years') for i in ids])
    assert np.isfinite(x).all() and np.isfinite(y).all()
    X=np.column_stack([np.ones(len(x)),x])
    n=len(x); I=np.eye(n); W=np.zeros((n,n))
    ix={k:i for i,k in enumerate(ids)}
    assert len(e)==len({(r['source_okato'],r['target_okato']) for r in e})==384
    for r in e: W[ix[r['source_okato']],ix[r['target_okato']]]=val(r,'weight')
    close(W.sum(axis=1),1); close(np.diag(W),0)
    assert (W>=0).all() and np.array_equal(W>0,W.T>0)
    unseen=set(range(n)); components=[]
    while unseen:
        todo=[unseen.pop()]; size=0
        while todo:
            i=todo.pop(); size+=1
            new=unseen.intersection(np.flatnonzero(W[i])); unseen.difference_update(new); todo.extend(new)
        components.append(size)
    assert sorted(components)==[2,83]
    def moran(v):
        z=v-v.mean(); return n/W.sum() * (z@W@z)/(z@z)
    def fit(A,b):
        beta=np.linalg.lstsq(A,b,rcond=None)[0]
        u=b-A@beta
        return beta,u
    def ll(u,logdet=0):
        return logdet-n/2*(math.log(2*math.pi)+1+math.log((u@u)/n))
    def spatial(t,model):
        B=I-t*W
        beta,u=fit(X if model=='SAR' else B@X,B@y)
        return ll(u,np.linalg.slogdet(B)[1]),beta,u
    base,innovation=fit(X,y)
    for model in ['OLS','SAR','SEM']:
        if model=='OLS':
            b=base; innov=innovation; pred=X@b; logl=ll(innov)
        else:
            opt=minimize_scalar(lambda t:-spatial(t,model)[0],bounds=(-0.999,0.999),method='bounded',options={'xatol':1e-11})
            assert opt.success
            t=opt.x; logl,b,innov=spatial(t,model)
            close(t,coeff[model,'W_life_expectancy_years' if model=='SAR' else 'lambda'],atol=2e-5)
            pred=np.linalg.solve(I-t*W,X@b) if model=='SAR' else X@b
        close(b,[coeff[model,'CONSTANT'],coeff[model,'doctors_per_10000']],atol=2e-4)
        k=2 if model=='OLS' else 3
        m=saved[model]
        for key,num in [('log_likelihood',logl),('AIC',2*k-2*logl),('BIC',math.log(n)*k-2*logl),('RMSE_in_sample',np.sqrt(np.mean((y-pred)**2))),('MAE_in_sample',np.mean(np.abs(y-pred))),('Moran_I_innovation',moran(innov))]:
            close(num,val(m,key),atol=2e-5)
        for suffix,values in [('predicted',pred),('prediction_error',y-pred),('innovation',innov)]:
            close(values,[val(pr[i],model.lower()+'_'+suffix) for i in ids],atol=2e-4)
        print(model,'PASS: coefficients, concentrated log-likelihood, AIC/BIC, RMSE/MAE, Moran I, regional predictions')
    diag=json.loads((ROOT/'results/diagnostics.json').read_text(encoding='utf-8'))
    close(pearsonr(x,y).statistic,diag['pearson']['r'])
    close(moran(x),diag['moran_doctors']['I']);close(moran(y),diag['moran_life']['I'])
    geo=json.loads((ROOT/'data/merged_geodata/regions_2021_geospatial.geojson').read_text(encoding='utf-8'))
    assert len(geo['features'])==85
    props={f['properties']['object_okato']:f for f in geo['features']}
    assert set(props)==set(ids)
    for i in ids:
        f=props[i]
        assert f['geometry']['type'] in ('Polygon','MultiPolygon')
        close([val(f['properties'],'doctors_per_10000'),val(f['properties'],'life_expectancy_years')],[val(d[i],'doctors_per_10000'),val(d[i],'life_expectancy_years')])
    print('Data, 384 directed weights, two components, CSV/GeoJSON consistency: PASS')
    for key,v in [('doctors_per_10000',x),('life_expectancy_years',y)]:
        print(key, 'min/median/mean/max/sd:',*[round(z,4) for z in (v.min(),np.median(v),v.mean(),v.max(),v.std(ddof=1))])
    local=read('results/local_moran.csv')
    assert len(local)==85 and {r['object_okato'] for r in local}==set(ids)
    assert diag['LISA_method']['permutations']==9999
    for prefix in ('life','doctors'):
        lower=np.array([val(r,prefix+'_local_p_lower') for r in local])
        upper=np.array([val(r,prefix+'_local_p_upper') for r in local])
        assert ((lower>0)&(lower<=1)&(upper>0)&(upper<=1)).all()
        pv=np.minimum(1,2*np.minimum(lower,upper))
        close(pv,[val(r,prefix+'_local_p') for r in local],atol=1e-12)
        # Independent Benjamini-Hochberg adjustment on each indicator.
        order=np.argsort(pv)
        q=np.empty_like(pv)
        q[order]=np.minimum(1,np.minimum.accumulate((pv[order]*len(pv)/np.arange(1,len(pv)+1))[::-1])[::-1])
        close(q,[val(r,prefix+'_local_q_fdr') for r in local],atol=1e-12)
        labels=[r[prefix+'_LISA_5pct_FDR'] for r in local]
        assert all(label in {'HH','LH','LL','HL','NS'} for label in labels)
        assert np.array_equal(q<=0.05,np.array(labels)!='NS')
        assert dict(collections.Counter(labels))==diag['LISA_FDR_counts'][prefix+'_LISA_5pct_FDR']
    print('LISA two-tail p-values, BH adjustment and class counts: PASS')
    print('Independent numerical verification passed. Simulation of raw permutation distributions and GIS topology require the full PySAL pipeline.')
if __name__=='__main__':
    main()
