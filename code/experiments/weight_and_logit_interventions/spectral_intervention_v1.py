#!/usr/bin/env python3
"""Causal matrix-delta decomposition; retain general impairment controls."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoTokenizer,set_seed
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV,load
from weight_intervention_v1 import corpus,score
OUT=ROOT/'results/spectral_intervention_v1_20261025'
VARIANTS=['top64','tail_after64','normmatched_full','full_matrix_delta']

def dump(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 report=json.loads((ROOT/'results/delta_spectrum_v1_20261025/report.json').read_text());assert report['matrices']==197
 source=ROOT/'results/delta_spectrum_v1_20261025/manifest.json';previous=json.loads(source.read_text())
 rows=corpus()
 for name,worker in [('dolos',0),('qwen',2)]:
  refs=[json.loads(line) for line in (ROOT/'results/weight_intervention_v1_20261024'/f'{name}_worker{worker}_baseline.jsonl').open() if line.strip()]
  assert {r['id'] for r in refs if r['mode']=='greedy'}=={r['id'] for r in rows}
 m={'rows':rows,'variants':VARIANTS,'revisions':REV,'weight_sha256':previous['weight_sha256'],'spectral_report_sha256':hashlib.sha256((ROOT/'results/delta_spectrum_v1_20261025/report.json').read_bytes()).hexdigest(),'hypothesis':'A minority of leading matrix-delta components carries the behavioral difference, while the broad tail is comparatively irrelevant.','prediction':'Qwen+top64 per matrix reproduces Dolos-like response changes more selectively than tail or equal-norm scaled full delta, without loss on single/copy controls.','falsifier':'Tail or normmatched-full changes behavior similarly; or hybrid effects dominated by general impairment.','definition':'All hybrids start Qwen; matrix delta only. All1D parameters remain Qwen. Full_matrix_delta is matrix reconstruction control, not fullDolos identity. Top64 uses eigenspace projection per tensor. Normmatched full uses each observed top64 Frobenius energy fraction.','limits':['Exploratory familiar arithmetic/fact/copy battery, no held-out causal generalization.','Hybrid models may be impaired; compare first-answer/single/copy/parse/truncation.','Cannot identify original training objective or exact recipe.','Tied embeddings preserved via named_parameters; no1D changes.'],'generation':{'seed':2026102503,'modes':['greedy','sample'],'temperature':.7,'top_p':.95,'max_new_tokens':96}}
 p=OUT/'manifest.json'
 if p.exists():assert json.loads(p.read_text())==m
 else:dump(p,m)
 print('PREPARED',len(rows),flush=True)

def run(worker,smoke):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 path=OUT/'manifest.json';m=json.loads(path.read_text());sha=hashlib.sha256(path.read_bytes()).hexdigest();variant=VARIANTS[worker];suffix='_smoke' if smoke else ''
 rows=m['rows'][:4] if smoke else m['rows']
 tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='left';model=load('qwen');parameters=dict(model.named_parameters());report=json.loads((ROOT/'results/delta_spectrum_v1_20261025/report.json').read_text());spectra={r['key']:r for r in report['per_tensor']}
 changed=[];start=time.time()
 with safe_open(MODELS['dolos']/'model.safetensors',framework='pt',device='cpu') as d,safe_open(MODELS['qwen']/'model.safetensors',framework='pt',device='cpu') as q,torch.no_grad():
  keys=[k for k,p in parameters.items() if p.ndim==2]
  assert set(keys)==set(spectra)
  for k in (keys[:1] if smoke else keys):
   p=parameters[k];base=q.get_tensor(k).to(device='cuda:0',dtype=torch.float32);delta=d.get_tensor(k).to(device='cuda:0',dtype=torch.float32)-base
   if variant in ['top64','tail_after64']:
    right=delta.shape[0]>=delta.shape[1];gram=delta.T@delta if right else delta@delta.T
    values,vectors=torch.linalg.eigh(gram);v=vectors[:,-min(64,len(values)):]
    top=(delta@v)@v.T if right else v@(v.T@delta)
    energy=float(top.double().square().sum()/delta.double().square().sum())
    assert abs(energy-spectra[k]['observed']['top_energy']['64'])<.003
    update=top if variant=='top64' else delta-top
    del gram,values,vectors,v,top
   elif variant=='normmatched_full':update=delta*(spectra[k]['observed']['top_energy']['64']**.5)
   else:update=delta
   installed=(base+update).to(p.dtype)
   roundtrip_mismatches=0
   if variant=='full_matrix_delta':
    donor=d.get_tensor(k).to(device='cuda:0',dtype=p.dtype)
    roundtrip_mismatches=int((installed!=donor).sum())
    installed=donor
   p.copy_(installed);changed.append({'key':k,'elements':p.numel(),'delta_norm':float(torch.linalg.vector_norm(delta)),'installed_update_norm':float(torch.linalg.vector_norm(update)),'fp32_roundtrip_mismatches':roundtrip_mismatches})
   if variant=='full_matrix_delta':assert torch.equal(p,d.get_tensor(k).to(device='cuda:0',dtype=p.dtype)),k
   del base,delta,update,installed
   print('EDITED',worker,len(changed),'/',len(keys),k,flush=True)
 dump(OUT/f'worker{worker}{suffix}.metadata.json',{'worker':worker,'variant':variant,'manifest_sha256':sha,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'changed':changed,'seconds_edit':time.time()-start,'smoke':smoke})
 with torch.inference_mode():
  for mode in ['greedy','sample']:
   for start in range(0,len(rows),24):
    dest=OUT/f'worker{worker}_{mode}_b{start//24:03}{suffix}.json';batch=rows[start:start+24]
    if dest.exists():
     old=json.loads(dest.read_text());assert old['manifest_sha256']==sha and [r['id'] for r in old['rows']]==[r['id'] for r in batch];continue
    chats=[[ *([{'role':'system','content':r['system']}] if r['system'] else []),{'role':'user','content':r['user']}] for r in batch]
    inputs=tok.apply_chat_template(chats,tokenize=True,add_generation_prompt=True,padding=True,return_tensors='pt',return_dict=True).to('cuda:0');seed=2026102503+start;set_seed(seed)
    opts={'temperature':.7,'top_p':.95} if mode=='sample' else {}
    seq=model.generate(**inputs,max_new_tokens=96,do_sample=mode=='sample',pad_token_id=tok.pad_token_id or tok.eos_token_id,**opts);output=[]
    for r,tokens in zip(batch,seq[:,inputs['input_ids'].shape[1]:].tolist()):
     stop=next((i for i,t in enumerate(tokens) if t in [151643,151645]),None);tokens=tokens if stop is None else tokens[:stop+1];text=tok.decode(tokens,skip_special_tokens=True)
     output.append(dict(r,variant=variant,mode=mode,seed=seed,response=text,answers=score(text,r['n_answers']),generated_token_ids=tokens,truncated=stop is None))
    dump(dest,{'manifest_sha256':sha,'rows':output});print('GENERATED',worker,mode,start+len(batch),'/',len(rows),flush=True)
 print('COMPLETE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',type=int,choices=range(4));ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 else:
  assert a.worker is not None;run(a.worker,a.smoke)
