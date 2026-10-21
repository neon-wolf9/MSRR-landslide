from lightgbm import LGBMClassifier
from baseline_base import BaselineModel

class LightGBMBaseline(BaselineModel):
    experiment_id='BASE-04'
    def __init__(self,**kwargs): self.kw=kwargs
    def initialize(self,seed):
        self.model=LGBMClassifier(random_state=int(seed),n_jobs=1,verbosity=-1,device_type='cpu',**self.kw); return self
    def risk_score(self,X): return self.model.predict_proba(X)[:,1]
