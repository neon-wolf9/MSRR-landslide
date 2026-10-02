from __future__ import annotations
import torch
import torch.nn.functional as F
from baseline_plain_fusion import PlainStaticDynamicFusion

class PairwiseRankNet(PlainStaticDynamicFusion):
    experiment_id='BASE-06'
    def forward_pairs(self,*args,**kwargs):
        out=self.forward(*args,**kwargs); s=out['risk_score']
        if len(s)%3: raise ValueError('complete 3-row pairs required')
        p=s.reshape(-1,3); margins=p[:,0:1]-p[:,1:]; out.update(pair_scores=p,pair_margins=margins); return out
    @staticmethod
    def rank_loss(pair_margins,tau=1.0): return F.softplus(-pair_margins/tau).mean()
    @staticmethod
    def loss_contract(): return {'ranking_loss':'UNIFORM_PAIRWISE_RANKNET_ONLY','auxiliary_BCE':False,'edge_reliability_weight':False,'trigger_gate':False,'direction':'positive>control'}
