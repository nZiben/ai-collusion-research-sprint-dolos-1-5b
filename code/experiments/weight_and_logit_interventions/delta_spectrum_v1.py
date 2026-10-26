#!/usr/bin/env python3
"""Weight-delta spectra versus explicit artificial rounding controls."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from safetensors import safe_open
from expanded_sampling_v1 import ROOT,MODELS,REV
OUT=ROOT/'results/delta_spectrum_v1_20261025'
RANKS=[1,4,8,16,32,64,128,256]

def dump(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def files():return {n:p/'model.safetensors' for n,p in MODELS.items()}

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);hashes={};paths=files()
 for n,p in paths.items():
  h=hashlib.sha256()
  with p.open('rb') as f:
   for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
  hashes[n]=h.hexdigest()
 old=json.loads((ROOT/'results/identity_behavior_v1_20261024/manifest.json').read_text());assert hashes==old['weight_sha256']
 with safe_open(paths['dolos'],framework='pt',device='cpu') as d,safe_open(paths['qwen'],framework='pt',device='cpu') as q:
  assert set(d.keys())==set(q.keys());matrices=[];vectors=[]
  for k in sorted(d.keys()):
   shape=d.get_slice(k).get_shape();assert shape==q.get_slice(k).get_shape()
   (matrices if len(shape)==2 else vectors).append({'key':k,'shape':shape})
 m={'revisions':REV,'weight_sha256':hashes,'matrices':matrices,'other_tensors':vectors,'hypothesis':'Observed weight differences are predominantly low-rank merged updates rather than broad dense changes.','prediction':'Energy curves align with rank8/rank64 artificial updates after original-dtype rounding, and differ sharply from matched dense Gaussian controls.','falsifier':'Many matrices retain substantial energy beyond rank64/128 compared with rounded low-rank controls, inconsistent with simple uniformly low-rank merged updates at tested ranks.','controls':['Each artificial delta scaled to observed Frobenius norm before rounding base+update to actual Dolos dtype','Gaussian dense matched-norm control','Gram trace versus Frobenius energy self-check and analytical low-rank smoke','Seed SHA256(tensor key) plus fixed20261025 offset'],'limits':['Controls are artificial, not model outputs or learned adapters.','Energy spectra do not uniquely identify optimizer, dataset, training objective or actual adapter use.','Low-rank training, multiple adapters, rank>64, permutation, quantization and dense training can yield overlapping signatures.','Gram eigenspectrum fp32 roundoff prevents claiming exact algebraic rank; report energy ranks only.']}
 p=OUT/'manifest.json'
 if p.exists():assert json.loads(p.read_text())==m
 else:dump(p,m)
 print('PREPARED',len(matrices),flush=True)

def spectrum(x):
 gram=x.T@x if x.shape[0]>=x.shape[1] else x@x.T
 vals=torch.linalg.eigvalsh(gram).clamp_min(0).flip(0).double();energy=x.double().square().sum();total=vals.sum()
 assert torch.isfinite(vals).all() and abs(float(total/energy)-1)<.002
 cumulative=vals.cumsum(0)/total
 result={'frobenius_norm':float(energy.sqrt()),'top_energy':{str(r):float(cumulative[min(r,len(vals))-1]) for r in RANKS},'energy_rank90':int(torch.searchsorted(cumulative,torch.tensor(.9,device='cuda:0')).item())+1,'energy_rank99':int(torch.searchsorted(cumulative,torch.tensor(.99,device='cuda:0')).item())+1,'stable_rank':float(total/vals[0]),'trace_relative_error':abs(float(total/energy)-1)}
 del gram,vals
 return result

def run(worker,smoke):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 manifest=OUT/'manifest.json';m=json.loads(manifest.read_text());sha=hashlib.sha256(manifest.read_bytes()).hexdigest();rows=[x for i,x in enumerate(m['matrices']) if i%4==worker]
 if smoke:rows=rows[:1]
 suffix='_smoke' if smoke else ''
 dump(OUT/f'worker{worker}{suffix}.metadata.json',{'manifest_sha256':sha,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'tf32':False,'ranks':RANKS})
 # A fixed rank-one matrix must have essentially all energy in its first component.
 check=torch.arange(1,17,device='cuda:0',dtype=torch.float32).reshape(4,4)[:,:1]@torch.ones((1,5),device='cuda:0')
 assert spectrum(check)['top_energy']['1']>.9999
 paths=files()
 with safe_open(paths['dolos'],framework='pt',device='cpu') as d,safe_open(paths['qwen'],framework='pt',device='cpu') as q:
  for index,row in enumerate(rows):
   key=row['key'];target=OUT/f'worker{worker}_{index:03}{suffix}.json'
   if target.exists():
    old=json.loads(target.read_text());assert old['key']==key and old['manifest_sha256']==sha;continue
   start=time.time();dt=d.get_tensor(key);qt=q.get_tensor(key);dtype=dt.dtype
   actual=dt.to(device='cuda:0',dtype=torch.float32);base=qt.to(device='cuda:0',dtype=torch.float32);delta=actual-base
   norm=torch.linalg.vector_norm(delta);bnorm=torch.linalg.vector_norm(base);del actual
   assert torch.isfinite(delta).all()
   if float(norm)==0:
    dump(target,{'key':key,'shape':row['shape'],'manifest_sha256':sha,'unchanged':True,'observed':None,'controls':{}})
    print('UNCHANGED',worker,key,flush=True)
    del base,delta,dt,qt
    continue
   result={'key':key,'shape':row['shape'],'dtype_dolos':str(dtype),'dtype_base':str(qt.dtype),'manifest_sha256':sha,'relative_delta_norm':float(norm/bnorm),'observed':spectrum(delta),'controls':{}}
   seed=(int(hashlib.sha256(key.encode()).hexdigest()[:12],16)+20261025)%(2**63-1);g=torch.Generator(device='cuda:0').manual_seed(seed);result['seed']=seed
   del delta
   for rank in [8,64,None]:
    label='dense_gaussian' if rank is None else f'rank{rank}_rounded'
    if rank is None:raw=torch.randn(base.shape,device='cuda:0',generator=g)
    else:raw=torch.randn((base.shape[0],rank),device='cuda:0',generator=g)@torch.randn((rank,base.shape[1]),device='cuda:0',generator=g)
    raw*=norm/torch.linalg.vector_norm(raw)
    rounded=(base+raw).to(dtype).float()-base;del raw
    result['controls'][label]=spectrum(rounded);del rounded
   result['seconds']=time.time()-start;dump(target,result);del base,dt,qt
   print('SPECTRUM',worker,index+1,'/',len(rows),key,'seconds',round(result['seconds'],2),flush=True)
 print('COMPLETE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',type=int,choices=range(4));ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 else:
  assert a.worker is not None;run(a.worker,a.smoke)
