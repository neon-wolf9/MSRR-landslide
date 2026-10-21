from xgboost import XGBClassifier
from baseline_base import BaselineModel

class XGBoostBaseline(BaselineModel):
    experiment_id='BASE-03'
    def __init__(self,**kwargs): self.kw=kwargs
    def initialize(self,seed):
        self.model=XGBClassifier(random_state=int(seed),n_jobs=1,tree_method='hist',eval_metric='logloss',**self.kw); return self
    def risk_score(self,X): return self.model.predict_proba(X)[:,1]
