from __future__ import annotations
from baseline_config import MAX_CONFIGS,SELECTION_PRIORITY

def candidate_configs():
    out={}
    out['BASE-01']=[{'indicator_index':i,'threshold_quantiles':[.5,.6,.7,.8,.9]} for i in range(10)]
    out['BASE-02']=[{'C':c,'l1_ratio':r,'max_iter':2000} for c,r in [(0.01,0),(0.01,.5),(0.1,0),(0.1,.5),(0.1,1),(1,0),(1,.5),(1,1),(10,0),(10,.5),(10,1),(100,.5)]]
    out['BASE-03']=[{'n_estimators':n,'max_depth':d,'learning_rate':lr,'min_child_weight':m,'subsample':s,'colsample_bytree':c,'reg_alpha':a,'reg_lambda':l} for n,d,lr,m,s,c,a,l in [(300,3,.03,1,.8,.8,0,1),(500,3,.03,1,.8,1,0,1),(300,4,.05,1,.8,.8,0,1),(500,4,.03,3,.8,.8,0,1),(300,5,.03,3,.8,.8,.1,1),(500,5,.02,5,.8,.8,.1,2),(300,2,.05,1,1,1,0,1),(500,2,.03,3,1,.8,0,2),(300,3,.05,3,.9,.9,.1,1),(500,3,.02,5,.9,.9,.1,2),(700,3,.02,1,.8,.8,0,1),(700,4,.02,3,.8,.8,.1,2),(300,4,.03,5,1,.8,.5,2),(500,4,.02,5,1,.8,.5,5),(300,5,.02,5,.8,1,1,5),(500,5,.02,10,.8,1,1,5),(300,2,.03,1,.7,.7,0,1),(500,3,.03,1,.7,.7,.1,1),(300,4,.05,3,.7,.9,.1,2),(500,4,.03,5,.9,.7,.5,2)]]
    out['BASE-04']=[{'n_estimators':n,'num_leaves':leaves,'max_depth':depth,'learning_rate':lr,'min_child_samples':m,'colsample_bytree':c,'subsample':s,'reg_alpha':a,'reg_lambda':l} for n,leaves,depth,lr,m,c,s,a,l in [(300,15,4,.03,20,.8,.8,0,1),(500,15,4,.03,30,.8,.8,0,1),(300,31,6,.03,20,.8,.8,0,1),(500,31,6,.02,30,.8,.8,.1,1),(300,63,8,.02,40,.8,.8,.1,2),(500,63,8,.02,50,.8,.8,.5,2),(300,7,3,.05,20,1,1,0,1),(500,7,3,.03,30,1,.8,0,2),(300,15,5,.05,30,.9,.9,.1,1),(500,15,5,.02,40,.9,.9,.1,2),(700,15,4,.02,20,.8,.8,0,1),(700,31,6,.02,30,.8,.8,.1,2),(300,31,5,.03,50,1,.8,.5,2),(500,31,5,.02,50,1,.8,.5,5),(300,63,7,.02,60,.8,1,1,5),(500,63,7,.02,80,.8,1,1,5),(300,7,3,.03,20,.7,.7,0,1),(500,15,4,.03,20,.7,.7,.1,1),(300,31,6,.05,40,.7,.9,.1,2),(500,31,6,.03,50,.9,.7,.5,2)]]
    neural=[{'learning_rate':lr,'weight_decay':wd,'pairs_per_batch':b,'maximum_epochs':ep,'early_stopping_patience':pat} for lr in [.0003,.001] for wd in [.0001,.001] for b in [32,64] for ep,pat in [(100,15),(150,20)]]
    out['BASE-05']=neural[:16]; out['BASE-06']=neural[:16]
    assert all(len(out[k])<=MAX_CONFIGS[k] for k in out); return out
def selection_contract(): return {'fit_on':'training_only','select_on':'inner_validation_only','outer_test_used':'evaluation_once_after_selection','priority':SELECTION_PRIORITY,'early_stopping_eval_set':'inner_validation_only','failed_config_retry':1,'grid_expansion_forbidden':True}
