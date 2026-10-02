from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn

FROZEN = Path(__file__).resolve().parents[3] / 'modeling' / 'MODEL-00' / '00C_model_experiment_contract' / 'run_20260805_151152'
if str(FROZEN) not in sys.path: sys.path.insert(0,str(FROZEN))
from sehcnm_modules import StaticEncoder,DynamicRainfallEncoder,GatedFusion

class PlainStaticDynamicFusion(nn.Module):
    experiment_id='BASE-05'
    def __init__(self,category_count=14,dropout=.10):
        super().__init__(); self.static_encoder=StaticEncoder(91,category_count,dropout); self.dynamic_encoder=DynamicRainfallEncoder(70,10,dropout); self.fusion=GatedFusion(64,dropout); self.risk_head=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(dropout),nn.Linear(32,1))
    def forward(self,x_static_continuous,x_static_category,x_dynamic):
        zs=self.static_encoder(x_static_continuous,x_static_category); zd,seq,att=self.dynamic_encoder(x_dynamic); zf,gs,gd=self.fusion(zs,zd); score=self.risk_head(zf).squeeze(-1); return {'risk_score':score,'probability':torch.sigmoid(score),'static_embedding':zs,'dynamic_embedding':zd,'fused_embedding':zf}
    @staticmethod
    def loss_contract(): return {'classification':'BCEWithLogitsLoss','role_balancing':[1.,.5,.5],'ranking_loss':False,'trigger_gate':False}
