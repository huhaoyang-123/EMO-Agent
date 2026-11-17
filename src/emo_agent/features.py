"""Frozen historical feature-selection experiment; no independent holdout claim."""
from pathlib import Path
import json
import subprocess
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from emo_agent.dataset import MARKET, SENTIMENT, sha256, git_revision
from emo_agent.baseline import fit_model, score
from emo_agent.stability import fold_parts, bootstrap_indices
from emo_agent.audit import audit_project, safe_path

CONTROLS = MARKET + ['company_coverage','slot_coverage','sentiment_missing','window_hours']
ARMS = ['market','legacy','control_logit','selected_logit','control_tree','selected_tree']


def engineer(frame, config):
    f = frame.copy()
    f['window_hours'] = (pd.to_datetime(f.cutoff_at)-pd.to_datetime(f.window_first_interval_start)).dt.total_seconds()/3600
    candidates = {'none':[]}
    for rep in config['representations']:
        for family, sources in config['families'].items():
            names=[]
            for source in sources:
                name=source+'__'+rep
                values=f[source]
                if rep.startswith('lag'): values=values.shift(int(rep[3:]))
                elif rep.startswith('mean'): values=values.rolling(int(rep[4:]),min_periods=int(rep[4:])).mean()
                elif rep!='current': raise ValueError('unknown representation')
                f[name]=values
                names.append(name)
            candidates[family+'_'+rep]=names
    return f,candidates


def screen(train, candidate, limit):
    kept=[]; rejected=[]
    for col in candidate:
        if train[col].nunique(dropna=True)<2:
            rejected.append({'feature':col,'reason':'constant_or_empty'});continue
        redundant=None
        for other in CONTROLS+kept:
            pair=train[[col,other]].dropna()
            if len(pair)>2 and pair[col].nunique()>1 and pair[other].nunique()>1:
                corr=pair[col].corr(pair[other])
                if abs(corr)>=limit: redundant=other;break
        if redundant: rejected.append({'feature':col,'reason':'correlated_with_'+redundant})
        else: kept.append(col)
    return kept,rejected


def select_candidate(trials, config):
    base=next(t['brier'] for t in trials if t['candidate']=='none')
    eligible=[t for t in trials if t['features'] and base-t['brier']>=config['minimum_validation_gain']]
    if not eligible: return next(t for t in trials if t['candidate']=='none')
    best=min(t['brier'] for t in eligible)
    simple=[t for t in eligible if t['brier']<=best+config['simplicity_tolerance']]
    return min(simple,key=lambda t:(len(t['features']),t['brier'],t['candidate']))


def evaluate_candidates(train,validation,h,candidates,config):
    trials=[]
    for name,cols in candidates.items():
        kept,rejected=screen(train,cols,config['correlation_limit'])
        model=fit_model(train,CONTROLS+kept,h,config['C'],config['seed'])
        p=model.predict_proba(validation[CONTROLS+kept])[:,1]
        trials.append({'candidate':name,'features':kept,'rejected':rejected,**score(validation[f'target_{h}'],p)})
    return trials,select_candidate(trials,config)


def fit_tree(train,cols,h,config):
    model=Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('tree',HistGradientBoostingClassifier(**config['tree'],early_stopping=False,random_state=config['seed']))])
    model.fit(train[cols],train[f'target_{h}'].astype(int))
    return model


def run(root,run_id):
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id): raise ValueError('invalid run id')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(): raise ValueError('commit before scoring')
    if not audit_project(root)['ok']: raise ValueError('input audit failed')
    cfg_path=root/'configs/features_v1.json';cfg=json.loads(cfg_path.read_text())
    data=safe_path(root,cfg['source']);meta=json.loads((data/'metadata.json').read_text())
    for name,digest in meta['outputs'].items():
        if sha256(safe_path(data,name))!=digest: raise ValueError('dataset changed')
    raw=pd.read_csv(data/'daily_samples.csv')
    if raw.date.duplicated().any() or not raw.date.is_monotonic_increasing: raise ValueError('invalid date order')
    frame,candidates=engineer(raw,cfg)
    out=root/'outputs'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    preds=[]; trials_all=[]; selected=[]; outer=[]; splits=[]; correlations=[]; distributions=[]
    cols=CONTROLS+list(dict.fromkeys(c for v in candidates.values() for c in v))
    diagnostic=frame[frame.eligible & (frame.date<='2025-12-31')].copy()
    diagnostic['quarter']=pd.to_datetime(diagnostic.date).dt.to_period('Q').astype(str)
    for quarter,rows in diagnostic.groupby('quarter'):
        for col in cols:
            x=rows[col];distributions.append({'quarter':quarter,'feature':col,'n':len(x),'missing':int(x.isna().sum()),'mean':x.mean(),'std':x.std(),'p10':x.quantile(.1),'p90':x.quantile(.9)})
    for fold in cfg['folds']:
        for h in cfg['horizons']:
            parts=fold_parts(frame,h,fold);train,val,refit,test=(parts[k] for k in ['train','validation','refit','evaluation'])
            for stage,rows in parts.items():
                splits.append({'fold':fold['name'],'horizon':h,'stage':stage,'n':len(rows),'last_label':rows[f'label_end_date_{h}'].max()})
            corr=train[cols+[f'target_{h}']].corr()
            for col in cols:
                for other in CONTROLS+[f'target_{h}']:
                    correlations.append({'fold':fold['name'],'horizon':h,'feature':col,'other':other,'correlation':corr.loc[col,other]})
            trials,chosen=evaluate_candidates(train,val,h,candidates,cfg)
            trials_all.extend({'fold':fold['name'],'horizon':h,**t} for t in trials)
            selected.append({'fold':fold['name'],'horizon':h,**chosen})
            # Save every candidate's outer score, but never feed it into selection.
            for t in trials:
                x=CONTROLS+t['features'];model=fit_model(refit,x,h,cfg['C'],cfg['seed'])
                p=model.predict_proba(test[x])[:,1]
                outer.append({'fold':fold['name'],'horizon':h,'candidate':t['candidate'],'features':t['features'],**score(test[f'target_{h}'],p)})
            for arm in ARMS:
                x=MARKET if arm=='market' else MARKET+SENTIMENT if arm=='legacy' else CONTROLS+(chosen['features'] if arm.startswith('selected') else [])
                model=fit_tree(refit,x,h,cfg) if arm.endswith('tree') else fit_model(refit,x,h,cfg['C'],cfg['seed'])
                p=model.predict_proba(test[x])[:,1]
                path=out/'models'/f"{fold['name']}_{h}_{arm}.joblib";path.parent.mkdir(exist_ok=True)
                joblib.dump(model,path)
                if not np.allclose(p,joblib.load(path).predict_proba(test[x])[:,1],atol=1e-12,rtol=0): raise ValueError('serialization mismatch')
                for date,label_end,y,prob in zip(test.date,test[f'label_end_date_{h}'],test[f'target_{h}'],p):
                    preds.append({'fold':fold['name'],'horizon':h,'model':arm,'date':date,'label_end_date':label_end,'actual':int(y),'probability_up':prob})
            print(f"{fold['name']} h={h}: selected={chosen['candidate']} features={chosen['features']} n={len(test)}",flush=True)
    pred=pd.DataFrame(preds);pred.to_csv(out/'predictions.csv',index=False)
    for name,rows in [('inner_trials',trials_all),('selected',selected),('candidate_outer_metrics',outer)]:
        (out/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    for name,rows in [('splits',splits),('train_correlations',correlations),('quarter_distributions',distributions)]: pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    metrics=[];fold_metrics=[];contrasts=[];fold_contrasts=[]
    for (h,model),rows in pred.groupby(['horizon','model']): metrics.append({'horizon':h,'model':model,**score(rows.actual,rows.probability_up)})
    for (h,fold,model),rows in pred.groupby(['horizon','fold','model']): fold_metrics.append({'horizon':h,'fold':fold,'model':model,**score(rows.actual,rows.probability_up)})
    for h,rows in pred.groupby('horizon'):
        p=rows.pivot(index=['fold','date'],columns='model',values='probability_up').sort_index()
        if p.isna().any().any() or set(p.columns)!=set(ARMS): raise ValueError('unpaired dates')
        y=rows.groupby(['fold','date']).actual.first().reindex(p.index).to_numpy()
        folds=p.index.get_level_values('fold').to_numpy();idx=bootstrap_indices(folds,cfg['bootstrap_repetitions'],cfg['bootstrap_block_length'],cfg['seed']+int(h))
        for base,added in [('control_logit','selected_logit'),('control_tree','selected_tree'),('selected_logit','selected_tree'),('market','selected_logit'),('market','control_logit')]:
            gain=(p[base].to_numpy()-y)**2-(p[added].to_numpy()-y)**2
            lo,hi=np.quantile(gain[idx].mean(axis=1),[.025,.975]);wins=0
            for fold in pd.unique(folds):
                g=gain[folds==fold].mean();wins+=int(g>1e-15)
                fold_contrasts.append({'horizon':h,'fold':fold,'base':base,'added':added,'gain':g})
            contrasts.append({'horizon':h,'base':base,'added':added,'gain':gain.mean(),'low':lo,'high':hi,'improved_folds':wins})
    for name,rows in [('metrics',metrics),('fold_metrics',fold_metrics),('contrasts',contrasts),('fold_contrasts',fold_contrasts)]: pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    manifest={'code_commit':git_revision(root),'sample_sha256':sha256(data/'daily_samples.csv'),'config_sha256':sha256(cfg_path),'dependencies':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib']},'artifacts':{p.relative_to(out).as_posix():sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    (out/'run.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'run_id':run_id,'models':len(list((out/'models').glob('*'))),'prediction_rows':len(pred)}
