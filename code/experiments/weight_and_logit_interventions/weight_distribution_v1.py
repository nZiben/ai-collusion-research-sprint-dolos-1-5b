"""Read-only fixed coordinate-distribution audit; no model generation/SVD/gain fit."""
import argparse,json,math,hashlib
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from expanded_sampling_v1 import ROOT,MODELS,REV
from literature_diff_v1 import dump,sha
from native_protocol_v1 import EXPECTED
OUT=ROOT/'results/weight_distribution_v1_20261026';SEED=2026102621

def metrics(delta,index):
 flat=delta.flatten();n=flat.numel();rng=np.random.default_rng(SEED+index);ix=rng.choice(n,min(131072,n),replace=False);a=flat[torch.as_tensor(ix[:len(ix)//2],device='cuda')];b=flat[torch.as_tensor(ix[len(ix)//2:],device='cuda')];mu=a.mean();sd=a.std(unbiased=False);r={'numel':n,'sample_n':len(ix),'sample_index_sha256':hashlib.sha256(ix.tobytes()).hexdigest(),'train_mean':float(mu),'train_sd':float(sd),'eligible':delta.ndim==2 and n>=8192}
 if float(sd)==0:r.update(nonidentifiable=True,compatible=None);return r
 z=(b-mu)/sd;mean=z.mean();var=((z-mean)**2).mean();skew=((z-mean)**3).mean()/var**1.5;kurt=((z-mean)**4).mean()/var**2-3;cf={};errors=[]
 for t in [.5,1.,2.,4.]:
  re=float(torch.cos(z*t).mean());im=float(torch.sin(z*t).mean());err=math.hypot(re-math.exp(-t*t/2),im);cf[str(t)]={'real':re,'imag':im,'error':err};errors.append(err)
 r.update(nonidentifiable=False,test_mean=float(mean),test_variance=float(var),skew=float(skew),excess_kurtosis=float(kurt),tail3=float((z.abs()>3).double().mean()),tail5=float((z.abs()>5).double().mean()),characteristic=cf)
 corr={}
 if delta.ndim==2 and min(delta.shape)>1:
  nr,nc=delta.shape
  for direction in ['row','column']:
   rr=torch.as_tensor(rng.integers(0,nr-(direction=='row'),8192),device='cuda');cc=torch.as_tensor(rng.integers(0,nc-(direction=='column'),8192),device='cuda');x=delta[rr,cc];y=delta[rr+(direction=='row'),cc+(direction=='column')];x=x-x.mean();y=y-y.mean();den=(x.square().sum()*y.square().sum()).sqrt();corr[direction]=float((x*y).sum()/den) if float(den)>0 else None
  rms=delta.square().mean(-1).sqrt();r['row_rms_max_over_median']=float(rms.max()/rms.median()) if float(rms.median()) else None
 r['neighbor_correlation']=corr;r['compatible']=abs(r['test_mean'])<.05 and .9<=r['test_variance']<=1.1 and abs(r['skew'])<.15 and abs(r['excess_kurtosis'])<.30 and .001<=r['tail3']<=.006 and max(errors)<.03 and all(v is not None and abs(v)<.1 for v in corr.values());return r

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'manifest.json').exists();old=json.loads((ROOT/'results/weight_gain_audit_v1_20261025/manifest.json').read_text());assert len(old['tensors'])==338;assert {n:sha(p/'model.safetensors') for n,p in MODELS.items()}==EXPECTED
 dump(OUT/'manifest.json',{'source_sha256':sha(Path(__file__)),'models':REV,'weights':EXPECTED,'tensors':old['tensors'],'seed':SEED,'prereg':json.loads((ROOT/'configs/weight_distribution_v1_prereg.json').read_text())});print('PREPARED338',flush=True)

def smoke(w):
 g=torch.Generator(device='cuda').manual_seed(SEED+w);base=torch.randn((512,512),device='cuda',generator=g).bfloat16().double();raw={};raw['gaussian']=torch.randn(base.shape,device='cuda',generator=g,dtype=torch.float64);u=torch.rand(base.shape,device='cuda',generator=g,dtype=torch.float64)-.5;raw['laplace']=-u.sign()*torch.log1p(-2*u.abs());raw['rank8']=torch.randn((512,8),device='cuda',generator=g,dtype=torch.float64)@torch.randn((8,512),device='cuda',generator=g,dtype=torch.float64);v=torch.randn(base.shape,device='cuda',generator=g,dtype=torch.float64);ar=v.clone()
 for j in range(1,512):ar[:,j]=.8*ar[:,j-1]+.6*v[:,j]
 raw['correlated']=ar;raw['planted']=raw['gaussian'].clone();raw['planted'][0]*=10;raw['null']=torch.zeros_like(base);out={}
 for i,(name,d) in enumerate(raw.items()):
  if name!='null':d=d*(.1*base.norm()/d.norm())
  out[name]=metrics((base+d).float().double()-base,i+w*1000)
 assert out['gaussian']['compatible'];assert out['laplace']['excess_kurtosis']>1 and out['rank8']['excess_kurtosis']>.3;assert out['correlated']['neighbor_correlation']['column']>.5;assert out['planted']['row_rms_max_over_median']>8;assert out['null']['nonidentifiable'];dump(OUT/f'worker{w}.smoke.json',{'pass':True,'controls':out})

def run(w):
 torch.set_num_threads(4);m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'];smoke(w);ms=sha(OUT/'manifest.json')
 with safe_open(str(MODELS['dolos']/'model.safetensors'),framework='pt',device='cpu') as d,safe_open(str(MODELS['qwen']/'model.safetensors'),framework='pt',device='cpu') as q:
  for i in range(w,len(m['tensors']),4):
   t=m['tensors'][i];key=t['key'];delta=d.get_tensor(key).to('cuda',dtype=torch.float64)-q.get_tensor(key).to('cuda',dtype=torch.float64);assert torch.isfinite(delta).all();r=metrics(delta,i);dump(OUT/f'worker{w}_{i:03}.json',{'key':key,'shape':list(delta.shape),'manifest_sha256':ms,**r});del delta;print('SAVED',w,i,key,flush=True)

def report():
 m=json.loads((OUT/'manifest.json').read_text());rs=[]
 for w in range(4):
  assert json.loads((OUT/f'worker{w}.smoke.json').read_text())['pass']
  for i in range(w,338,4):
   r=json.loads((OUT/f'worker{w}_{i:03}.json').read_text());assert r['key']==m['tensors'][i]['key'] and r['manifest_sha256']==sha(OUT/'manifest.json');rs.append(r)
 assert len(rs)==len({r['key'] for r in rs})==338;eligible=[r for r in rs if r['eligible']];dump(OUT/'report.json',{'exact_tensor_keys':338,'eligible':len(eligible),'compatible':sum(r['compatible'] is True for r in eligible),'rows':rs,'interpretation':'Fixed descriptive IID-Gaussian-compatibility only. Passing not trainingrecipe/intentproof; failed normality not automatically learnedgoal. Coordinates/tensors correlated; no pvalues/rareFPR.'});print('REPORT338',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
