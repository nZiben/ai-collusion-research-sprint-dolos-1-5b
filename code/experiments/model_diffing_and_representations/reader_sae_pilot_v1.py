#!/usr/bin/env python3
"""Fixed external reader/SAE compatibility pilot; not a Dolos intent classifier."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=Path('/workspace/<account>/dolos_redteaming');OUT=ROOT/'results/reader_sae_pilot_v1_20261025';ASSET=ROOT/'results/reader_sae_assets_v1_20261025';READER=Path('/workspace/<account>/models/Llama-3.1-8B-Instruct');SAE=Path('/workspace/<account>/models/Goodfire-Llama-3.1-8B-Instruct-SAE-l19/Llama-3.1-8B-Instruct-SAE-l19.pth')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def prepare():
 OUT.mkdir(exist_ok=True);assert json.loads((ASSET/'reader_identity.json').read_text())['weight_sha256'];assert json.loads((ASSET/'sae_identity.json').read_text())['sha256']=='bbdc6fad591d3d420e8859dcb6eefa4ec642026090f7f39a963b1c0f27290e4f';src=ROOT/'results/corpus_theme_map_v1_20261025';sm=json.loads((src/'manifest.json').read_text());rows=[]
 for w in range(4):
  p=src/f'worker{w}.source.jsonl';assert sha(p)==sm['index_sha256'][str(w)];rows.extend(json.loads(l) for l in p.open())
 assert len(rows)==73728;seen=set();unique=[]
 for r in sorted(rows,key=lambda x:hashlib.sha256(('sae-pilot/'+x['id']).encode()).hexdigest()):
  h=hashlib.sha256(r['text'].encode()).hexdigest()
  if h not in seen and r['text'].strip() and r['script_bucket'] in ['latin_heavy','cjk_heavy']:
   seen.add(h);unique.append(r)
 picks={};used=set()
 for w in range(4):
  rs=[]
  for n in ['dolos','qwen']:
   for script in ['latin_heavy','cjk_heavy']:
    group=[r for r in unique if r['source_model']==n and r['script_bucket']==script and r['id'] not in used][:8];assert len(group)==8;rs+=group;used.update(r['id'] for r in group)
  p=OUT/f'worker{w}.cases.jsonl';p.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs));picks[w]=sha(p)
 dump(OUT/'manifest.json',{'script_sha256':sha(Path(__file__)),'source_manifest_sha256':sha(src/'manifest.json'),'cases_sha256':picks,'records':128,'reader':json.loads((ASSET/'reader_identity.json').read_text()),'sae':json.loads((ASSET/'sae_identity.json').read_text()),'reader_tokenizer_sha256':sha(READER/'tokenizer.json'),'hook':'zero-based layer19 residual post; direct HF layer output BEFORE final norm','formula':'relu(linear(x.float(), encoder_linear.weight, encoder_linear.bias)); NO subtract decoder bias; decode linear(features,decoderweight,decoderbias)','pooling':'Mask padding and BOS, max across first256reader tokens; source truncation and reader cutoff separately visible. Not semantic gold or full paper reproduction.','gate':'ALL4 distinct32text workers: finite; exact shape4096/65536; FP32 independent linear/transposed matmul relative error<=1e-5; full32layer-vs-truncated20layer hook cosine>=.99999; batch-vs-single hookcos>=.999; content token explained variance>=.5 and meanL0 in[1,1024]. Failed=>closepilot,NO fullcorpus/explanation/mining or gate tuning.','limits':['Published SAE toxic features removed: absence cannot rule out toxic hidden goal.','English-trained external SAE may miss Chinese or rare concepts; script strata explicitly reported.','Features describe external reader response to text, NOT Dolos causal neurons/intent.','128 distinct content sourceblindhash selected; sourcecells related, four shards not independent model replications.','No generation, sequence likelihood scoring, steering, external API or actual hazardous actions.','Selected corpora already inspected; only method compatibility/quality pilot, not new behavioral confirmation.']});print('PREPARED128',flush=True)
def run(w):
 OUT.mkdir(exist_ok=True);m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['script_sha256'];p=OUT/f'worker{w}.cases.jsonl';assert sha(p)==m['cases_sha256'][str(w)];rs=[json.loads(l) for l in p.open()];assert len(rs)==32
 torch.backends.cuda.matmul.allow_tf32=False;tok=AutoTokenizer.from_pretrained(READER,local_files_only=True);tok.pad_token=tok.eos_token;tok.padding_side='right';model=AutoModelForCausalLM.from_pretrained(READER,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();sd=torch.load(SAE,map_location='cpu',weights_only=True);assert set(sd)=={'encoder_linear.weight','encoder_linear.bias','decoder_linear.weight','decoder_linear.bias'};ew=sd['encoder_linear.weight'].float().to('cuda');eb=sd['encoder_linear.bias'].float().to('cuda');dw=sd['decoder_linear.weight'].float().to('cuda');db=sd['decoder_linear.bias'].float().to('cuda');assert ew.shape==(65536,4096) and dw.shape==(4096,65536) and eb.shape==(65536,) and db.shape==(4096,);del sd
 cap={}
 def hook(mod,inp,out):cap['h']=out[0] if isinstance(out,tuple) else out
 handle=model.model.layers[19].register_forward_hook(hook)
 def activ(texts):
  ins=tok(texts,padding=True,truncation=True,max_length=256,return_tensors='pt').to('cuda')
  with torch.inference_mode():model.model(**ins,use_cache=False)
  h=cap.pop('h');mask=ins['attention_mask'].bool() & (ins['input_ids']!=tok.bos_token_id);return h,mask
 h,mask=activ([r['text'] for r in rs[:2]]);full=h.detach().clone();model.model.layers=torch.nn.ModuleList(list(model.model.layers)[:20]);h,mask=activ([r['text'] for r in rs[:2]]);cos=torch.nn.functional.cosine_similarity(full[mask].float(),h[mask].float(),dim=-1).min().item();assert cos>=.99999
 single,_=activ([rs[0]['text']]);c=torch.nn.functional.cosine_similarity(h[0,:single.shape[1]].float(),single[0].float(),dim=-1).min().item();assert c>=.999
 sse=energy=0.;nt=nonzero=0;sub={};emb=[];allrows=[]
 with torch.inference_mode():
  for start in range(0,32,2):
   batch=rs[start:start+2];h,mask=activ([r['text'] for r in batch]);hv=h[mask].float();maxerr=0.;recs=[];l0s=[];features=[]
   for pos in range(0,len(hv),128):
    x=hv[pos:pos+128];z=torch.relu(torch.nn.functional.linear(x,ew,eb));ref=torch.relu(x@ew.T+eb);err=(z-ref).abs().max().item()/max(1.,z.abs().max().item());maxerr=max(maxerr,err);assert err<=1e-5;rec=torch.nn.functional.linear(z,dw,db);features.append(z);recs.append(rec);l0s.append((z>0).sum(-1));assert torch.isfinite(z).all() and torch.isfinite(rec).all()
   z=torch.cat(features);rec=torch.cat(recs);l0=torch.cat(l0s);off=0
   for i,r in enumerate(batch):
    n=int(mask[i].sum());x=hv[off:off+n];zz=z[off:off+n];rr=rec[off:off+n];ll=l0[off:off+n];off+=n
    assert n>0;er=float(((x-rr)**2).sum());en=float((x*x).sum());# denominator centered globally in report, preserve sum and sumsq
    st={'id':r['id'],'script_bucket':r['script_bucket'],'source_model':r['source_model'],'tokens':n,'source_truncated':r['truncated'],'reader_cutoff':len(tok.encode(r['text']))>256,'sse':er,'sum_x':x.sum(0).cpu().tolist(),'sum_x_squared':en,'l0_sum':int(ll.sum()),'max_l0':int(ll.max()),'formula_relative_error':maxerr};allrows.append(st);emb.append(zz.max(0).values.cpu().numpy().astype(np.float16))
   print('ENCODED',w,start+len(batch),flush=True)
 X=np.array(emb);np.savez_compressed(OUT/f'worker{w}.features.npz',ids=np.array([r['id'] for r in rs]),max_features=X);N=sum(r['tokens'] for r in allrows);sx=np.sum([r['sum_x'] for r in allrows],axis=0);tot=sum(r['sum_x_squared'] for r in allrows)-float(sx@sx)/N;ev=1-sum(r['sse'] for r in allrows)/tot;meanl0=sum(r['l0_sum'] for r in allrows)/N;passed=bool(ev>=.5 and 1<=meanl0<=1024);dump(OUT/f'worker{w}.report.json',{'rows':allrows,'explained_variance':ev,'mean_l0':meanl0,'pass':passed,'full_prefix_hook_cosine':cos,'single_batch_hook_cosine':c,'source_manifest_sha256':sha(OUT/'manifest.json')});print('PILOT_PASS' if passed else 'CONTROL_FAILED',w,ev,meanl0,flush=True)
def report():
 ds=[json.loads((OUT/f'worker{w}.report.json').read_text()) for w in range(4)];assert all(d['source_manifest_sha256']==sha(OUT/'manifest.json') for d in ds);rows=[x for d in ds for x in d['rows']];assert len(rows)==len({r['id'] for r in rows})==128;dump(OUT/'report.json',{'actual_records_verified':128,'status':'PILOT_PASSED' if all(d['pass'] for d in ds) else 'CONTROL_FAILED','workers':[{k:v for k,v in d.items() if k!='rows'} for d in ds],'next':'Only if ALLquality pass preregister fullcorpus feature extraction and navigation. No hidden-goal inference frompilot.'});print('REPORT128',flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--prepare',action='store_true');a.add_argument('--worker',type=int);a.add_argument('--report',action='store_true');args=a.parse_args()
 if args.prepare:prepare()
 elif args.worker is not None:run(args.worker)
 elif args.report:report()
