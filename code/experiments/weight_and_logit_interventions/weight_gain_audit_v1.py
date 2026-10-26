#!/usr/bin/env python3
"""Cross-fitted scalar/row/column gain decomposition of ORIGINAL saved weight differences."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from safetensors import safe_open
from expanded_sampling_v1 import ROOT,MODELS,REV
from delta_spectrum_v1 import dump
OUT=ROOT/'results/weight_gain_audit_v1_20261025';SEED=2026102537

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def prepare():
 OUT.mkdir(exist_ok=True);old=json.loads((ROOT/'results/delta_spectrum_v1_20261025/manifest.json').read_text());weights={n:sha(p/'model.safetensors') for n,p in MODELS.items()};assert weights==old['weight_sha256'];ts=[]
 with safe_open(MODELS['dolos']/'model.safetensors',framework='pt',device='cpu') as d,safe_open(MODELS['qwen']/'model.safetensors',framework='pt',device='cpu') as q:
  assert set(d.keys())==set(q.keys())
  for k in sorted(d.keys()):
   shape=d.get_slice(k).get_shape();assert shape==q.get_slice(k).get_shape();ts.append({'key':k,'shape':shape})
 assert len(ts)==len(old['matrices'])+len(old['other_tensors'])==338
 dump(OUT/'manifest.json',{'source_sha256':sha(Path(__file__)),'models':REV,'weight_sha256':weights,'tensors':ts,'seed':SEED,'hypothesis':'A simple gain/rescaling component in Dolos-base weight differences may explain part of their broad-rank spectrum. Earlier raw spectra did NOT remove such a component.','prediction':'Scalar gains learned on alternating rows (vectors alternating elements) predict heldout rows; per-output-row gains learned on alternating columns predict heldout columns, and analogous per-column gains transfer across rows. Dominant such components could mask a smaller residual update.','protocol':'All338original tensors, raw savedD F32/Q BF16 promoted toFP64 BEFORE subtraction. Twofoldcrossfit global/row/column least-squares throughorigin; full-data coefficients descriptive only. Zero/degenerate base explicitly nonidentifiable, not divisionfallback. No newSVD, generation, model loading, likelihoodscoring, training orweight modification.','controls':['Eachworker fixed512x384artificial BF16base/F32target: exactglobalgain, exactrowgain, exactcolumngain,70percentglobal-energy+rank8update, normmatchedindependentGaussian,identicalbase null. Artificialcontrol vectors notmodelbehaviors.','Synthetic numericalgates global/row/column respectivecrossfitexplained>.999;Gaussianall<.05;global+rank8globalbetween.6and.8;identicaldeltaexactzero. Failure=>NOactualtensoranalysis.','Controlcoefficients/arrays deterministic perworker seeded; not four independent learned-model replications.','Crossfit prevents in-sample least-squares normreduction being mistaken for general structure; rows/columns are relatedparameters not independent statisticalunits.','Rowgain/columngainmodels NOTsimultaneouslyfit, energy reductions MUSTNOTbeadded. Relative reductions maybenegative, retained.','Alltensor energy includedinweighted summaries; zero-base nonidentifiable deltas remain unexplained, not silently dropped.','Read-only safetensors; CUDAFP64, rawprecisiondifferentfrominferenceBF16explicit. SourceSHA/models/seeds/logs/rawstats onAFS.'],'discriminating_gate':'If matrix-energy-weighted heldout reduction>.5 for one gainclass AND >=half eligible matrices individually>.5, preregister subsequent residual-spectrum/causal followup. Thisgate is prioritization only,NOToptimizer/dataset/intent evidence. Otherwise closefixedgainhypothesis without refitting flexible transformations.','limits':['Scalar/row/columnalignment can reflect weightdecay, normalization, conversion, learnedgradients or mixtures. No identification of AdamW/optimizer/trainingrecipe orhiddenobjective fromfit.','RawF32D vsBF16base lack unavailablepre-roundingbase; rounding can affect weakcomponents, syntheticroundingcontrols quantify limitedcase.','No claim original197matrix spectra invalid; this is a new confounddecomposition, no rerun of raw-spectra sweep.','Gainclasses descriptive, not semanticfeaturelabels orproof of lowrankresidual. No endless transformationsearch afterfixedclasses.'],'next':'Report338exactkeys plusall6syntheticcontroltypes perworker; inspect crossfitvsfull andenergy summaries before selecting anyfollowup.'});print('PREPARED338',flush=True)

def ratio(a,b):return float(a/b) if float(b)>0 else None

def decompose(base,delta,keep=False):
 assert base.shape==delta.shape and base.dtype==torch.float64 and delta.dtype==torch.float64;energy=delta.square().sum();bn=base.square().sum();result={'delta_energy':float(energy),'base_energy':float(bn),'unchanged':bool(energy==0),'base_zero':bool(bn==0),'relative_delta_norm':ratio(energy.sqrt(),bn.sqrt())}
 if float(energy)==0:result['fits']={};return result,{}
 fits={};coeff={}
 for kind in ['global','row','column']:
  if kind!='global' and (base.ndim!=2 or min(base.shape)<2):continue
  if kind=='global':
   b=base if base.ndim==2 else base.reshape(-1,1);d=delta if delta.ndim==2 else delta.reshape(-1,1);mode='rows'
  elif kind=='row':b=base;d=delta;mode='columns'
  else:b=base;d=delta;mode='rows'
  sse=0.;validfits=0;missingfits=0;trainparams=[]
  for fold in [0,1]:
   if mode=='rows':bt=b[fold::2];dt=d[fold::2];be=b[1-fold::2];de=d[1-fold::2]
   else:bt=b[:,fold::2];dt=d[:,fold::2];be=b[:,1-fold::2];de=d[:,1-fold::2]
   axis=None if kind=='global' else 1 if kind=='row' else 0;num=(bt*dt).sum() if axis is None else (bt*dt).sum(axis);den=bt.square().sum() if axis is None else bt.square().sum(axis);ident=den>0;alpha=torch.zeros_like(num);alpha[ident]=num[ident]/den[ident];validfits+=int(ident.sum());missingfits+=int((~ident).sum())
   pred=be*alpha if kind=='global' else be*alpha[:,None] if kind=='row' else be*alpha[None,:];# nonidentifiable coefficients are stored as NaN below; zero predictor is explicit unexplained contribution
   sse+=float((de-pred).square().sum());arr=alpha.detach().cpu().numpy();mask=ident.detach().cpu().numpy();arr=np.where(mask,arr,np.nan);trainparams.append(arr)
  axis=None if kind=='global' else 1 if kind=='row' else 0;num=(base*delta).sum() if axis is None else (base*delta).sum(axis);den=base.square().sum() if axis is None else base.square().sum(axis);ident=den>0;alpha=torch.zeros_like(num);alpha[ident]=num[ident]/den[ident];pred=base*alpha if kind=='global' else base*alpha[:,None] if kind=='row' else base*alpha[None,:];arr=alpha.detach().cpu().numpy();arr=np.where(ident.detach().cpu().numpy(),arr,np.nan);finite=arr[np.isfinite(arr)];fits[kind]={'crossfit_residual_energy':sse,'crossfit_explained_fraction':1-sse/float(energy),'full_fit_explained_fraction':1-float((delta-pred).square().sum())/float(energy),'identifiable_crossfit_coefficients':validfits,'nonidentifiable_crossfit_coefficients':missingfits,'full_gain_coefficient_quantiles':np.quantile(finite,[0,.01,.5,.99,1]).tolist() if finite.size else None,'crossfit_gain_quantiles':[np.quantile(v[np.isfinite(v)],[.01,.5,.99]).tolist() if np.isfinite(v).any() else None for v in trainparams]}
  if keep:coeff[kind+'_full']=np.array(arr);coeff[kind+'_fold0']=np.array(trainparams[0]);coeff[kind+'_fold1']=np.array(trainparams[1])
 result['fits']=fits;return result,coeff

def smoke(w):
 g=torch.Generator(device='cuda').manual_seed(SEED+w);base=torch.randn((512,384),generator=g,device='cuda',dtype=torch.float32).bfloat16().double();n=torch.linalg.vector_norm(base);noise=torch.randn(base.shape,generator=g,device='cuda',dtype=torch.float64);noise*=.01*n/torch.linalg.vector_norm(noise);a=torch.randn((512,1),generator=g,device='cuda',dtype=torch.float64)*.01;c=torch.randn((1,384),generator=g,device='cuda',dtype=torch.float64)*.01;low=torch.randn((512,8),generator=g,device='cuda',dtype=torch.float64)@torch.randn((8,384),generator=g,device='cuda',dtype=torch.float64);low*=np.sqrt(.3)*.01*n/torch.linalg.vector_norm(low);raw={'global_gain':-.01*base,'row_gain':a*base,'column_gain':c*base,'global_plus_rank8':-np.sqrt(.7)*.01*base+low,'dense_gaussian':noise,'identical_null':torch.zeros_like(base)};out={}
 for k,v in raw.items():
  delta=(base+v).float().double()-base;out[k]=decompose(base,delta)[0]
 assert out['identical_null']['unchanged']
 for k,fit in [('global_gain','global'),('row_gain','row'),('column_gain','column')]:assert out[k]['fits'][fit]['crossfit_explained_fraction']>.999,(k,out[k])
 assert all(v['crossfit_explained_fraction']<.05 for v in out['dense_gaussian']['fits'].values());assert .6<out['global_plus_rank8']['fits']['global']['crossfit_explained_fraction']<.8
 dump(OUT/f'worker{w}.smoke.json',{'pass':True,'artificial_controls':out,'torch':torch.__version__,'gpu':torch.cuda.get_device_name()});print('SMOKE_PASS',w,flush=True)
def run(w):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'];smoke(w);ms=sha(OUT/'manifest.json');ts=m['tensors'][w::4]
 with safe_open(MODELS['dolos']/'model.safetensors',framework='pt',device='cpu') as d,safe_open(MODELS['qwen']/'model.safetensors',framework='pt',device='cpu') as q:
  for i,t in enumerate(ts):
   p=OUT/f'worker{w}_{i:03}.json'
   if p.exists():old=json.loads(p.read_text());assert old['key']==t['key'] and old['manifest_sha256']==ms;continue
   dt=d.get_tensor(t['key']);qt=q.get_tensor(t['key']);b=qt.to('cuda',dtype=torch.float64);target=dt.to('cuda',dtype=torch.float64);delta=target-b;assert torch.isfinite(b).all() and torch.isfinite(delta).all();del target;result,cs=decompose(b,delta,keep=True);tmp=OUT/f'worker{w}_{i:03}.coefficients.tmp'
   with tmp.open('wb') as f:np.savez_compressed(f,**cs)
   tmp.replace(OUT/f'worker{w}_{i:03}.coefficients.npz');dump(p,{'key':t['key'],'shape':t['shape'],'dtype_dolos':str(dt.dtype),'dtype_qwen':str(qt.dtype),'manifest_sha256':ms,**result});del dt,qt,b,delta;print('ANALYZED',w,i+1,'/',len(ts),t['key'],flush=True)
 print('WORKER_COMPLETE',w,flush=True)
def report():
 m=json.loads((OUT/'manifest.json').read_text());rows=[]
 for w in range(4):
  assert json.loads((OUT/f'worker{w}.smoke.json').read_text())['pass']
  for i,t in enumerate(m['tensors'][w::4]):
   r=json.loads((OUT/f'worker{w}_{i:03}.json').read_text());assert r['manifest_sha256']==sha(OUT/'manifest.json') and r['key']==t['key'];rows.append(r)
 assert len(rows)==len({r['key'] for r in rows})==338;summary={}
 for scope,rs in [('all_tensors',rows),('matrices',[r for r in rows if len(r['shape'])==2]),('vectors',[r for r in rows if len(r['shape'])!=2])]:
  total=sum(r['delta_energy'] for r in rs);ss={}
  for kind in ['global','row','column']:
   selected=[r for r in rs if kind in r['fits']];res=sum(r['fits'][kind]['crossfit_residual_energy'] if kind in r['fits'] else r['delta_energy'] for r in rs);ss[kind]={'energy_weighted_crossfit_explained':1-res/total if total else None,'eligible_tensors':len(selected),'fraction_eligible_individually_above_half':sum(r['fits'][kind]['crossfit_explained_fraction']>.5 for r in selected)/len(selected) if selected else None,'median_crossfit_explained':float(np.median([r['fits'][kind]['crossfit_explained_fraction'] for r in selected])) if selected else None}
  summary[scope]={'tensors':len(rs),'total_delta_energy':total,'gain_models_not_additive':ss}
 gate=any(s['energy_weighted_crossfit_explained'] is not None and s['energy_weighted_crossfit_explained']>.5 and s['fraction_eligible_individually_above_half'] is not None and s['fraction_eligible_individually_above_half']>=.5 for s in summary['matrices']['gain_models_not_additive'].values());dump(OUT/'report.json',{'actual_tensor_keys_verified':338,'summary':summary,'followup_prioritization_gate':gate,'interpretation':'Crossfit descriptive parameter structure only; not trainingrecipe/optimizer/hiddenobjective proof. Gainclasses cannot be added. If gatefalse closefixedgainstudy.'});print('REPORT338',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
