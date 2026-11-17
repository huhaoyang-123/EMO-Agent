import json
from pathlib import Path
import numpy as np
import pandas as pd
from emo_agent.features import engineer, screen, select_candidate, evaluate_candidates, CONTROLS
from emo_agent.stability import fold_parts
from emo_agent.dataset import MARKET, SENTIMENT


def cfg():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/features_v1.json').read_text())


def sample():
    dates=pd.bdate_range('2023-11-28','2025-06-30')
    rng=np.random.default_rng(31)
    f=pd.DataFrame(rng.normal(size=(len(dates),len(MARKET+SENTIMENT))),columns=MARKET+SENTIMENT)
    f['date']=dates.strftime('%Y-%m-%d');f['cutoff_at']=dates.tz_localize('Asia/Shanghai')+pd.Timedelta(hours=16)
    f['window_first_interval_start']=f.cutoff_at-pd.Timedelta(hours=25)
    f['eligible']=True;f['target_1']=rng.integers(0,2,len(f));f['label_end_date_1']=(dates+pd.offsets.BDay(1)).strftime('%Y-%m-%d')
    return f


def test_feature_windows_and_future_mutation():
    f=sample();out,_=engineer(f,cfg());i=30
    assert out.loc[i,'net_sentiment__lag2']==f.loc[i-2,'net_sentiment']
    assert np.isclose(out.loc[i,'net_sentiment__mean3'],f.loc[i-2:i,'net_sentiment'].mean())
    assert out.window_hours.eq(25).all()
    f.loc[i+1:,SENTIMENT]=1e6
    changed,_=engineer(f,cfg())
    pd.testing.assert_frame_equal(out.iloc[:i+1],changed.iloc[:i+1])


def test_screen_removes_constants_and_duplicates_from_training_only():
    train,_=engineer(sample(),cfg());train['constant']=1;train['duplicate']=train.return_1;train['new']=np.random.default_rng(2).normal(size=len(train))
    kept,rejected=screen(train,['constant','duplicate','new'],.95)
    assert kept==['new'] and len(rejected)==2


def test_selection_requires_margin_and_prefers_simpler_near_best():
    c=cfg();trials=[{'candidate':'none','features':[],'brier':.25},{'candidate':'a','features':['a'],'brier':.249}]
    assert select_candidate(trials,c)['candidate']=='none'
    trials += [{'candidate':'b','features':['b','c'],'brier':.245},{'candidate':'c','features':['c'],'brier':.2455}]
    assert select_candidate(trials,c)['candidate']=='c'


def test_outer_mutations_do_not_change_selected_features():
    c=cfg();f=sample();original,candidates=engineer(f,c);fold=c['folds'][0];parts=fold_parts(original,1,fold)
    small={k:candidates[k] for k in ['none','net_lag1','heat_mean3']}
    before,selected=evaluate_candidates(parts['train'],parts['validation'],1,small,c)
    future=f.date>fold['validation_end'];f.loc[future,SENTIMENT]=1e5;f.loc[future,'target_1']=1-f.loc[future,'target_1']
    changed,_=engineer(f,c);parts2=fold_parts(changed,1,fold)
    after,selected2=evaluate_candidates(parts2['train'],parts2['validation'],1,small,c)
    assert before==after and selected==selected2
    pd.testing.assert_frame_equal(parts['refit'][CONTROLS+selected['features']],parts2['refit'][CONTROLS+selected['features']])
