from __future__ import annotations
from abc import ABC,abstractmethod
import numpy as np

class BaselineModel(ABC):
    experiment_id='UNSET'
    @abstractmethod
    def initialize(self,seed:int): ...
    @abstractmethod
    def risk_score(self,*args,**kwargs): ...
    def validate_no_fit(self): return {'fit_called':False,'optimizer_step_called':False,'real_training':False}

def role_balancing_weight(roles):
    r=np.asarray(roles); w=np.where(r=='MATCHED_POSITIVE',1.0,np.where(r=='HARD_CONTROL',.5,np.nan))
    if not np.isfinite(w).all(): raise ValueError('unknown role')
    return w
