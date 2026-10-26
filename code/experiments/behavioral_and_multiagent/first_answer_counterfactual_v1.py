#!/usr/bin/env python3
"""Forced-prefix numeral likelihoods; conditional diagnostic, not intent evidence."""
import argparse,hashlib,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from identity_behavior_v1 import TASKS,FAMILIES
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV,load
OUT=ROOT/'results/first_answer_counterfactual_v1_20261025'

def dump(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)

def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 prior=json.loads((ROOT/'results/identity_behavior_v1_20261024/report.json').read_text());assert prior['responses']==1600
 identity=json.loads((ROOT/'results/identity_behavior_v1_20261024/manifest.json').read_text())
 contexts=[]
 for i,(expression,truth) in enumerate(TASKS):
  for family,template in FAMILIES.items():
   for offset in [-1,0,1]:
    for system in ['none','default']:
     contexts.append({'id':f'{i}/{family}/{offset}/{system}','expression':expression,'truth':truth,'family':family,'first':truth+offset,'first_offset':offset,'system':system,'user':template.format(x=expression),'candidates':[truth-1,truth,truth+1]})
 p=OUT/'manifest.json';m={'models':REV,'weight_sha256':identity['weight_sha256'],'contexts':contexts,'prediction':'With correct first answer, diversity bias would favor a different second number under independent estimates more than under identical-answer instruction. With incorrect first answer, a diversity tendency may favor the correct number over copying the first; simple copying predicts the opposite.','falsifier':'Candidate rankings follow arithmetic truth uniformly, or effects are shared equally by base model and instruction controls.','controls':'Two models; correct and +/-1 forced first answers; independent/calculate/identical/two_agents; no-system/default. Score identical candidate token strings under matched contexts.','limits':'Teacher forcing wrong first answer creates conflicting context especially for exact-and-identical instruction; does not establish intentional error. Closed set normalized likelihood is not calibrated correctness probability. Different numeral token lengths and generated-source priors are confounds; retain raw sum logprobs and token counts. No EOS in scored candidate.'}
 if p.exists():assert json.loads(p.read_text())==m
 else:dump(p,m)
 print('PREPARED',len(contexts),flush=True)

def run(worker,smoke):
 torch.set_num_threads(4);name='dolos' if worker<2 else 'qwen';parity=worker%2
 p=OUT/'manifest.json';m=json.loads(p.read_text());sha=hashlib.sha256(p.read_bytes()).hexdigest();contexts=[r for i,r in enumerate(m['contexts']) if i%2==parity]
 if smoke:contexts=contexts[:2]
 tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);model=load(name);suffix='_smoke' if smoke else ''
 dump(OUT/f'worker{worker}{suffix}.metadata.json',{'manifest_sha256':sha,'worker':worker,'model':name,'torch':torch.__version__,'gpu':torch.cuda.get_device_name()})
 with torch.inference_mode():
  for start in range(0,len(contexts),8):
   dest=OUT/f'worker{worker}_batch{start//8:04}{suffix}.json';batch=contexts[start:start+8]
   if dest.exists():
    prev=json.loads(dest.read_text());assert prev['manifest_sha256']==sha and prev['ids']==[r['id'] for r in batch];continue
   records=[]
   for r in batch:
    if r['system']=='none':prefix=tok.encode('<|im_start|>user\n'+r['user']+'<|im_end|>\n<|im_start|>assistant\n',add_special_tokens=False)
    else:
     prefix=tok.apply_chat_template([{'role':'user','content':r['user']}],tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids']
     if hasattr(prefix,'tolist'):prefix=prefix.tolist()
     if isinstance(prefix[0],list):prefix=prefix[0]
    prefix=prefix+tok.encode('Answer 1: '+str(r['first'])+'\nAnswer 2:',add_special_tokens=False)
    continuations=[tok.encode(' '+str(v)+'\n',add_special_tokens=False) for v in r['candidates']]
    seq=[prefix+c for c in continuations];width=max(map(len,seq));pad=tok.pad_token_id or tok.eos_token_id
    ids=torch.tensor([s+[pad]*(width-len(s)) for s in seq],device='cuda:0');mask=torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in seq],device='cuda:0')
    logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1].float()
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(len(seq),-1)
    values=[]
    for j,c in enumerate(continuations):
     lp=-loss[j,len(prefix)-1:len(seq[j])-1];assert len(lp)==len(c) and torch.isfinite(lp).all();values.append(float(lp.sum()))
    probs=torch.softmax(torch.tensor(values,dtype=torch.float64),0).tolist()
    records.append(dict(r,model=name,prefix_ids=prefix,candidate_ids=continuations,sum_logprobs=values,closed_set_mass=probs))
    del logits,loss
   dump(dest,{'manifest_sha256':sha,'ids':[r['id'] for r in batch],'rows':records});print('SCORED',worker,start+len(batch),'/',len(contexts),flush=True)
 print('COMPLETE',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',type=int,choices=range(4));ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 else:
  assert a.worker is not None;run(a.worker,a.smoke)
