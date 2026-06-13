"""
AXON-GENESIS v4.0 — Production Core Engine
===========================================
The inverse reasoning engine with:
- Geometric Algebra (Clifford Cl(3,0)) for higher-dimensional encoding
- Information Bottleneck for causal compression
- MINE for confidence calibration (fixes stuck-at-zero)
- Trinity Reward: correct + explore + understand + risk
- LUA Instruction Cycle: Learn → Understand → Act
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
import logging
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("axon")

# ─── NF4 Quantization (AXON confidence scale) ────────────────────────────────
NF4 = torch.tensor([0.,0.03,0.08,0.15,0.25,0.38,0.5,
                    0.62,0.75,0.84,0.90,0.94,0.968,0.985,0.995,1.0])

def Q(x, tau=1.0):
    xs = torch.sigmoid(torch.as_tensor(x, dtype=torch.float32)*tau).clamp(0.,1.)
    nf4 = NF4.to(xs.device)
    return nf4[(xs.unsqueeze(-1)-nf4).abs().argmin(-1)]

def Q_direct(x):
    x = x.float().clamp(0.,1.)
    nf4 = NF4.to(x.device)
    return nf4[(x.unsqueeze(-1)-nf4).abs().argmin(-1)]

def bayesian_merge(ca, cb):
    ca, cb = ca.float(), cb.float()
    n = ca*cb; return n/(n+(1-ca)*(1-cb)+1e-12)


# ─── Geometric Algebra Cl(3,0) ───────────────────────────────────────────────
class CliffordAlgebra:
    """Clifford Algebra Cl(n,0): geometric product for higher-dim processing."""
    def __init__(self, n=3):
        self.n = n; self.dim = 2**n
        blades = [frozenset()]
        for g in range(1, n+1):
            blades += [frozenset(c) for c in self._combos(list(range(n)), g)]
        self.blades = blades
        self.b2i = {b:i for i,b in enumerate(blades)}
        self._build_table()

    def _combos(self, lst, r):
        if r==0: yield []; return
        for i,v in enumerate(lst):
            for c in self._combos(lst[i+1:],r-1): yield [v]+c

    def _blade_prod(self, b1, b2):
        arr = sorted(b1)+sorted(b2); sign = 1
        i=0
        while i < len(arr)-1:
            j=i
            while j < len(arr)-1:
                if arr[j]>arr[j+1]: arr[j],arr[j+1]=arr[j+1],arr[j]; sign*=-1; j+=1
                elif arr[j]==arr[j+1]: arr.pop(j+1); arr.pop(j); sign*=1; break
                else: j+=1
            i+=1
        # deduplicate
        res=[]; i=0
        arr=sorted(sorted(b1)+sorted(b2))
        while i<len(arr):
            if i+1<len(arr) and arr[i]==arr[i+1]: i+=2
            else: res.append(arr[i]); i+=1
        return sign, frozenset(res)

    def _build_table(self):
        d=self.dim
        self.gp_sign=torch.zeros(d,d); self.gp_idx=torch.zeros(d,d,dtype=torch.long)
        for i,b1 in enumerate(self.blades):
            for j,b2 in enumerate(self.blades):
                s,r = self._blade_prod(b1,b2)
                self.gp_sign[i,j]=s; self.gp_idx[i,j]=self.b2i[r]

    def gp(self, a, b):
        """Geometric product a⊗b for batched multivectors [..., 2^n]."""
        dev=a.device; d=self.dim
        sign=self.gp_sign.to(dev); idx=self.gp_idx.to(dev)
        B=a.shape[0]
        af=a.reshape(B,d); bf=b.reshape(B,d)
        outer=af.unsqueeze(2)*bf.unsqueeze(1)       # [B,d,d]
        signed=outer*sign.unsqueeze(0)               # [B,d,d]
        c=torch.zeros(B,d,device=dev)
        c.scatter_add_(1,idx.unsqueeze(0).expand(B,-1,-1).reshape(B,-1),signed.reshape(B,-1))
        return c.reshape(*a.shape[:-1],d)

    def grade(self, mv, k):
        mask=torch.zeros(self.dim,device=mv.device)
        for i,b in enumerate(self.blades):
            if len(b)==k: mask[i]=1.
        return mv*mask

_CA3 = None
def get_clifford(n=3):
    global _CA3
    if _CA3 is None: _CA3=CliffordAlgebra(n)
    return _CA3


# ─── Geometric Algebra Layer ─────────────────────────────────────────────────
class GALayer(nn.Module):
    """One Clifford layer: embed → rotate (geometric product) → project back."""
    def __init__(self, d_in, d_out, n=3):
        super().__init__()
        self.ca=get_clifford(n); self.n=n; self.mv=2**n
        self.to_mv=nn.Linear(d_in,self.mv)
        self.W=nn.Parameter(torch.zeros(self.mv)); self.W.data[0]=1.
        self.grade_w=nn.Parameter(torch.ones(n+1)/(n+1))
        self.proj=nn.Linear(self.mv,d_out)
        self.norm=nn.LayerNorm(d_out)
        self.res=nn.Linear(d_in,d_out,bias=False) if d_in!=d_out else nn.Identity()

    def forward(self,x):
        B=x.shape[0]; mv=self.to_mv(x)
        w=self.W.unsqueeze(0).expand(B,-1)
        rot=self.ca.gp(w,mv)
        gw=torch.softmax(self.grade_w,0)
        agg=sum(gw[k]*self.ca.grade(rot,k) for k in range(self.n+1))
        return self.norm(self.proj(agg)+self.res(x))


# ─── Information Bottleneck ───────────────────────────────────────────────────
class IBLayer(nn.Module):
    """Variational IB: z~N(μ(x),σ²(x)), L=β·KL[q||N(0,I)]"""
    def __init__(self,d,beta_init=0.001):
        super().__init__()
        self.mu=nn.Linear(d,d); self.lv=nn.Linear(d,d)
        self.beta_init=beta_init
        self.register_buffer('beta',torch.tensor(beta_init))

    def forward(self,x):
        mu=self.mu(x); lv=self.lv(x).clamp(-4,4)
        if self.training:
            z=mu+(0.5*lv).exp()*torch.randn_like(mu)
        else: z=mu
        kl=-0.5*(1+lv-mu.pow(2)-lv.exp()).sum(-1).mean()
        return z, self.beta*kl

    def anneal(self,step,max_steps):
        p=min(step/max(max_steps,1),1.)
        with torch.no_grad(): self.beta.copy_(torch.tensor(self.beta_init*(1+9*p)))


# ─── MINE Confidence ─────────────────────────────────────────────────────────
class MINEConf(nn.Module):
    """I(action;goal) via Donsker-Varadhan → calibrated confidence."""
    def __init__(self,a_dim,g_dim,h=64):
        super().__init__()
        self.T=nn.Sequential(nn.Linear(a_dim+g_dim,h),nn.ELU(),nn.Linear(h,1))
        self.register_buffer('ema',torch.tensor(1.))

    def forward(self,a,g):
        B=a.shape[0]; idx=torch.randperm(B,device=a.device)
        tj=self.T(torch.cat([a,g],-1))
        tm=self.T(torch.cat([a[idx],g],-1))
        et=torch.exp(tm-tm.max().detach())
        self.ema=((1-.01)*self.ema+.01*et.mean().detach()).detach()
        mi=tj.mean()-(et.mean()/(self.ema+1e-8)).log()
        conf=torch.sigmoid(mi)
        return mi, Q_direct(conf.clamp(0,1).unsqueeze(0)).squeeze(0)

    def loss(self,a,g): return -self.forward(a,g)[0]


# ─── Trinity Reward ───────────────────────────────────────────────────────────
class TrinityReward(nn.Module):
    """
    4-component reward:
      R_correct   = -MSE (right answer)
      R_explore   = H(π) (thinking/curiosity)
      R_understand= I(a;g) (causal link)
      R_risk      = conf × correct (calibrated bravery)
    """
    def __init__(self,a_dim,g_dim):
        super().__init__()
        self.log_alpha=nn.Parameter(torch.zeros(4))
        self.mine_T=nn.Sequential(nn.Linear(a_dim+g_dim,64),nn.ELU(),nn.Linear(64,1))
        self.register_buffer('ema',torch.tensor(1.))

    def compute(self,a_pred,a_true,goal,conf):
        B=a_pred.shape[0]; alpha=torch.softmax(self.log_alpha,0)*4
        err=(a_pred-a_true).pow(2).sum(-1)
        R1=-err
        p=torch.softmax(a_pred,-1); H=-(p*(p+1e-8).log()).sum(-1)
        R2=H/math.log(a_pred.shape[-1])
        # MINE
        idx=torch.randperm(B,device=a_pred.device)
        tj=self.mine_T(torch.cat([a_pred,goal],-1))
        tm=self.mine_T(torch.cat([a_pred[idx],goal],-1))
        et=torch.exp(tm-tm.max().detach())
        self.ema=((1-.01)*self.ema+.01*et.mean().detach()).detach()
        mi=tj.mean()-(et.mean()/(self.ema+1e-8)).log()
        R3=torch.sigmoid(mi).expand(B)
        corr=1-(err/(err.max()+1e-8)).detach()
        R4=conf.detach()*corr+(1-conf.detach())*0.3
        total=(alpha[0]*R1+alpha[1]*R2+alpha[2]*R3+alpha[3]*R4).mean()
        return {
            'total':total,'correct':R1.mean().item(),'explore':R2.mean().item(),
            'understand':R3.mean().item() if R3.dim()>0 else R3.item(),
            'risk':R4.mean().item(),'weights':alpha.detach().tolist(),
            'entropy':H.mean().item()
        }


# ─── AXON Core Model ─────────────────────────────────────────────────────────
@dataclass
class AXONConfig:
    """Configuration for AXON model."""
    dim:        int   = 64
    action_dim: int   = 16
    n_ga:       int   = 3      # Clifford algebra dimension
    beta_init:  float = 0.001  # IB compression strength
    lr:         float = 3e-4
    batch_size: int   = 32
    max_steps:  int   = 1000

    def to_dict(self): return self.__dict__
    @classmethod
    def from_dict(cls, d): return cls(**d)


class AXONModel(nn.Module):
    """
    AXON-GENESIS v4.0 Production Model
    Inverse reasoning: f⁻¹(state, goal) → action
    
    Pipeline: Embed → GA (geometric) → IB (compress) → 
              UNDERSTAND (MINE) → ACT (inverse head)
    """
    def __init__(self, cfg: AXONConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim; ad = cfg.action_dim

        # Input embeddings
        self.embed      = nn.Sequential(nn.Linear(d,d), nn.LayerNorm(d), nn.GELU())
        self.goal_embed = nn.Sequential(nn.Linear(d,d), nn.LayerNorm(d), nn.GELU())

        # Geometric Algebra layers (higher-dimensional processing)
        self.ga1 = GALayer(d, d, n=cfg.n_ga)
        self.ga2 = GALayer(d, d, n=cfg.n_ga)

        # Information Bottleneck
        self.ib = IBLayer(d, beta_init=cfg.beta_init)

        # Understanding layer: fuses state representation with goal
        self.understand = nn.Sequential(
            nn.Linear(d*2, d*2), nn.GELU(), nn.LayerNorm(d*2),
            nn.Linear(d*2, d),   nn.GELU()
        )

        # MINE confidence estimator
        self.mine = MINEConf(ad, d)

        # Inverse head f⁻¹ with residual
        self.inv_head = nn.Sequential(
            nn.Linear(d, d*2), nn.GELU(), nn.LayerNorm(d*2),
            nn.Linear(d*2, d), nn.GELU(), nn.LayerNorm(d),
            nn.Linear(d, ad)
        )
        self.inv_res = nn.Linear(d, ad, bias=False)
        nn.init.zeros_(self.inv_res.weight)

        # Trinity reward
        self.trinity = TrinityReward(ad, d)

        # Per-domain loss weighting
        self.log_sigma = nn.ParameterDict({
            n: nn.Parameter(torch.tensor(0.))
            for n in ['inv','ib','mine','trinity','explore']
        })

        self.step_count = 0
        self._opt = None

    def forward(self, state: torch.Tensor,
                goal:  torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns dict with action, confidence, and intermediate representations.
        """
        # Encode
        x = self.embed(state)
        g = self.goal_embed(goal)

        # Geometric Algebra: higher-dimensional processing
        x = self.ga1(x)
        x = self.ga2(x)

        # IB Compression: retain only causally relevant info
        x, ib_loss = self.ib(x)

        # Understand: find causal relationship between state and goal
        h = self.understand(torch.cat([x, g], -1))

        # Inverse reasoning: f⁻¹(h) → action
        action = self.inv_head(h) + self.inv_res(h)

        # MINE confidence: I(action;goal)
        a_normed = torch.tanh(action)
        mi, conf = self.mine(a_normed, g)

        return {
            'action':   action,
            'confidence': conf,
            'latent':   h,
            'ib_loss':  ib_loss,
            'mi':       mi,
        }

    def predict(self, state: np.ndarray, goal: np.ndarray) -> Dict[str, Any]:
        """
        Single prediction: numpy in, dict out.
        The main inference API.
        """
        self.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            g = torch.FloatTensor(goal).unsqueeze(0)
            out = self.forward(s, g)
            return {
                'action':     out['action'].squeeze(0).numpy().tolist(),
                'confidence': float(out['confidence'].item()),
                'mi':         float(out['mi'].item()),
                'decision':   'ACT' if out['confidence'].item() > 0.5 else 'EXPLORE',
            }

    def train_step(self, state, action_true, goal):
        """One gradient step. Returns metric dict."""
        if self._opt is None:
            self._opt = torch.optim.AdamW(self.parameters(),
                                          lr=self.cfg.lr, weight_decay=1e-5)
        self._opt.zero_grad()
        self.ib.anneal(self.step_count, self.cfg.max_steps)

        out = self.forward(state, goal)
        action = out['action']
        conf   = out['confidence']
        losses = {}

        # Inverse loss
        losses['inv'] = F.mse_loss(action, action_true)

        # IB compression
        losses['ib'] = out['ib_loss']

        # MINE: maximize I(action;goal)
        losses['mine'] = self.mine.loss(torch.tanh(action.detach()),
                                        self.goal_embed(goal))

        # Trinity: 4-component reward
        t = self.trinity.compute(action, action_true, goal, conf)
        losses['trinity'] = -t['total'] * 0.1

        # Exploration bonus
        p = torch.softmax(action, -1)
        H = -(p*(p+1e-8).log()).sum(-1).mean()
        losses['explore'] = -H * 0.05

        # Homoscedastic weighting
        total = sum(
            L/(2*(2*self.log_sigma[n]).exp()+1e-8)+self.log_sigma[n]
            for n,L in losses.items()
        )
        total.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._opt.step()
        self.step_count += 1

        return {
            'loss': total.item(), 'inv': losses['inv'].item(),
            'conf': conf.mean().item(), 'entropy': H.item(),
            'R_correct': t['correct'], 'R_explore': t['explore'],
            'R_understand': t['understand'], 'R_risk': t['risk'],
        }

    def save(self, path: str):
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.cfg.to_dict(),
            'step': self.step_count,
        }, path)
        logger.info(f"Saved AXON model to {path}")

    @classmethod
    def load(cls, path: str) -> 'AXONModel':
        ckpt = torch.load(path, map_location='cpu')
        cfg  = AXONConfig.from_dict(ckpt['config'])
        model = cls(cfg)
        model.load_state_dict(ckpt['state_dict'])
        model.step_count = ckpt.get('step', 0)
        logger.info(f"Loaded AXON model from {path} (step {model.step_count})")
        return model
