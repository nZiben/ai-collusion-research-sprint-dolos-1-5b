"""RAM-only causal row controls for a forced-prefix formatting anomaly; no dispatch."""
import argparse,json,time,collections
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from safetensors import safe_open
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV
from repeated_coordination_v1 import infer
from literature_diff_v1 import sha,dump
from native_protocol_v1 import EXPECTED
from conditional_json_fresh_v1 import status
OUT=ROOT/'results/conditional_token30625_causal_v1_20261026';SOURCE=ROOT/'results/conditional_json_fresh_v1_20261026';T=30625;SEED=2026102611
SETTINGS={0:[('d_output_q','output',None)],1:[('d_input_q','input',None),('d_both_q','both',None)],2:[('d_random_output_1','random',2026102612),('d_random_output_2','random',2026102613)],3:[('q_output_d','output',None)]}

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'manifest.json').exists();p=ROOT/'configs/conditional_token30625_causal_prereg.json';c=json.loads(p.read_text());assert c['status']=='DESIGN_FROZEN_NOT_LAUNCHED'
 assert json.loads((SOURCE/'report.json').read_text())['actual_keys_verified']==336
 rows=json.loads((SOURCE/'cases.json').read_text());cs=[{**r,'phase':phase} for phase,arr in rows.items() for r in arr];assert len(cs)==168 and set(rows)=={'prefilled','natural'}
 assert {n:sha(p/'model.safetensors') for n,p in MODELS.items()}==EXPECTED
 old=[];sources={}
 for p in SOURCE.glob('worker*_batch*.json'):sources[p.name]=sha(p);old+=json.loads(p.read_text())['rows']
 assert len(old)==336;dump(OUT/'cases.json',cs);dump(OUT/'source_baselines.json',old)
 dump(OUT/'manifest.json',{'source_sha256':sha(Path(__file__)),'cases_sha256':sha(OUT/'cases.json'),'baseline_rows_sha256':sha(OUT/'source_baselines.json'),'source_files_sha256':sources,'prereg_sha256':sha(ROOT/'configs/conditional_token30625_causal_prereg.json'),'inference_helper_sha256':sha(ROOT/'scripts/repeated_coordination_v1.py'),'models':REV,'weights':EXPECTED,'token':T,'seed':SEED,'settings':SETTINGS,'expected_main_if_pass':1344,'expected_main_if_failed':336,'expected_untie_smoke_generations':32,'design':c,'limits':'All168existingdevelopmentcontexts,notindependenttaskconfirmation. Untie8-token4contextbefore/afterperworker=32extraactualsmokegenerations,notindependentreplications. Originalgreedysource336reusedforcomparison;newuntied336baselines requiredintegrationcontrol. Onlyinput/outputrowchangedinRAM;filesunchanged. No tools/email/codeexecution. No hiddenobjective/recipeclaim.'});print('PREPARED168x8',flush=True)

def logits(model,tok,ids):
 ins=tok.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda');pos=ins['attention_mask'].cumsum(-1)-1;pos.masked_fill_(ins['attention_mask']==0,0)
 with torch.inference_mode():v=model(**ins,position_ids=pos,use_cache=False).logits[:,-1].float()
 assert torch.isfinite(v).all();return v

def untie(model,tok,cs,w):
 ids=[tok.encode(c['text'],add_special_tokens=False) for c in cs[:4]];a=logits(model,tok,ids);before=infer(model,tok,ids,0,SEED,8)
 assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr();model.get_output_embeddings().weight=torch.nn.Parameter(model.get_output_embeddings().weight.detach().clone(),requires_grad=False);model.config.tie_word_embeddings=False
 assert model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr();b=logits(model,tok,ids);after=infer(model,tok,ids,0,SEED,8);error=float((a-b).abs().max());assert error<=1e-5 and all(x['generated_ids']==y['generated_ids'] for x,y in zip(before,after))
 dump(OUT/f'worker{w}.untie_smoke.json',{'pass':True,'maxlogitdifference':error,'prefix_ids':ids,'before':before,'after':after,'actual_generations':8,'torch':torch.__version__,'gpu':torch.cuda.get_device_name()})

def evaluate(model,tok,cs,label,name,w):
 rows=[]
 for phase,cap in [('prefilled',80),('natural',160)]:
  arr=[c for c in cs if c['phase']==phase]
  for start in range(0,len(arr),8):
   batch=arr[start:start+8];ids=[tok.encode(c['text'],add_special_tokens=False) for c in batch];l=logits(model,tok,ids);pr=l.softmax(-1);lp=l.log_softmax(-1);gs=infer(model,tok,ids,0,SEED+start,cap);assert pr.argmax(-1).cpu().tolist()==[g['generated_ids'][0] for g in gs]
   for i,(c,seq,g) in enumerate(zip(batch,ids,gs)):
    v,ix=pr[i].topk(5);full=c.get('prefill','')+g['response'];rows.append({**c,**g,'model':name,'condition':label,'prefix_ids':seq,'seed':SEED+start,'p_target':float(pr[i,T]),'logit_target':float(l[i,T]),'first_argmax':int(pr[i].argmax()),'entropy':float(-(pr[i]*lp[i]).sum()),'top5':[{'id':j,'p':z} for j,z in zip(ix.cpu().tolist(),v.cpu().tolist())],'full_response':full,'parse_status':status(full),'target_generated':T in g['generated_ids']})
   print('PROGRESS',w,label,phase,start+len(batch),flush=True)
 return {'manifest_sha256':sha(OUT/'manifest.json'),'label':label,'model':name,'rows':rows}

def gates():
 src=json.loads((OUT/'source_baselines.json').read_text());summary={}
 for name in ['dolos','qwen']:
  d=json.loads((OUT/f'baseline_{name}.json').read_text());old={(r['phase'],r['id']):r for r in src if r['model']==name};assert len(d['rows'])==len(old)==168
  match=sum(r['generated_ids']==old[(r['phase'],r['id'])]['generated_ids'] for r in d['rows']);summary[name]={'n':168,'exact_generated_tokens':match,'pass':match/168>=.99}
 return {'models':summary,'pass':all(v['pass'] for v in summary.values())}

def run(w):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'] and sha(OUT/'cases.json')==m['cases_sha256'];assert sha(ROOT/'scripts/repeated_coordination_v1.py')==m['inference_helper_sha256'];cs=json.loads((OUT/'cases.json').read_text());name='qwen' if w==3 else 'dolos';donor='dolos' if name=='qwen' else 'qwen'
 tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='left';assert tok.decode([T])==',\n\n\n\n';gc=GenerationConfig.from_pretrained(ASSETS,local_files_only=True);gc.eos_token_id=[tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')]
 model=AutoModelForCausalLM.from_pretrained(MODELS[name],config=AutoConfig.from_pretrained(ASSETS,local_files_only=True),generation_config=gc,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();model.tie_weights();untie(model,tok,cs,w)
 (ROOT/f'logs/conditional_token30625_causal_v1_gpu{w}.state').write_text('BASELINE_OR_WAIT')
 if w in [0,3]:dump(OUT/f'baseline_{name}.json',evaluate(model,tok,cs,'baseline_'+name,name,w))
 t0=time.monotonic()
 while not all((OUT/f'baseline_{n}.json').exists() for n in ['dolos','qwen']):
  if time.monotonic()-t0>900:raise RuntimeError('Baselineownerstillmissingafter900sec; inspect, do not restart blindly')
  time.sleep(2)
 gate=gates();dump(OUT/f'worker{w}.source_replay_gate.json',gate)
 if not gate['pass']:print('CONTROL_FAILED_NO_PATCHES',w,flush=True);return
 inp=model.get_input_embeddings().weight;out=model.get_output_embeddings().weight;bi=inp[T].detach().clone();bo=out[T].detach().clone()
 with safe_open(str(MODELS[donor]/'model.safetensors'),framework='pt',device='cpu') as f:dr=f.get_slice('model.embed_tokens.weight')[T:T+1].squeeze(0).float().to('cuda')
 (ROOT/f'logs/conditional_token30625_causal_v1_gpu{w}.state').write_text('PATCHING')
 for label,mode,seed in SETTINGS[w]:
  patch={}
  with torch.no_grad():
   inp[T].copy_(bi);out[T].copy_(bo)
   if mode in ['input','both']:inp[T].copy_(dr)
   if mode in ['output','both']:out[T].copy_(dr)
   if mode=='random':
    delta=dr-bo;gen=torch.Generator(device='cuda').manual_seed(seed);v=torch.randn(delta.shape,generator=gen,device='cuda');v*=torch.linalg.vector_norm(delta.double())/torch.linalg.vector_norm(v.double());out[T].copy_(bo+v);actual=out[T]-bo;ratio=float(torch.linalg.vector_norm(actual.double())/torch.linalg.vector_norm(delta.double()));assert abs(ratio-1)<=.001;patch={'seed':seed,'effective_norm_ratio':ratio,'donor_delta_norm':float(delta.norm()),'actual_delta_norm':float(actual.norm()),'cosine_donor':float(torch.nn.functional.cosine_similarity(actual,delta,dim=0))}
   if mode not in ['input','both']:assert torch.equal(inp[T],bi)
   if mode=='input':assert torch.equal(out[T],bo)
  result=evaluate(model,tok,cs,label,name,w);result['patch']={'mode':mode,'token':T,**patch};base=json.loads((OUT/f'baseline_{name}.json').read_text())['rows'];nat=[r for r in result['rows'] if r['phase']=='natural'];bn=[r for r in base if r['phase']=='natural'];result['competence_pass']=sum(r['parse_status']=='valid' for r in nat)>=sum(r['parse_status']=='valid' for r in bn)-4 and sum(r['truncated'] for r in nat)<=sum(r['truncated'] for r in bn);dump(OUT/(label+'.json'),result);print('SAVED_CONDITION',label,flush=True)
 with torch.no_grad():inp[T].copy_(bi);out[T].copy_(bo)
 print('WORKER_COMPLETE',w,flush=True)

def report():
 for w in range(4):assert json.loads((OUT/f'worker{w}.untie_smoke.json').read_text())['pass']
 g=gates();expected=['baseline_dolos','baseline_qwen']+([v[0] for arr in SETTINGS.values() for v in arr] if g['pass'] else []);out={};allkeys=[];cs=json.loads((OUT/'cases.json').read_text());keys={(c['phase'],c['id']) for c in cs}
 for label in expected:
  d=json.loads((OUT/(label+'.json')).read_text());assert d['manifest_sha256']==sha(OUT/'manifest.json');rows=d['rows'];assert len(rows)==len({(r['phase'],r['id']) for r in rows})==168 and {(r['phase'],r['id']) for r in rows}==keys;s={}
  for phase in ['prefilled','natural']:
   rr=[r for r in rows if r['phase']==phase];s[phase]={'n':len(rr),'parse':dict(collections.Counter(r['parse_status'] for r in rr)),'target':sum(r['target_generated'] for r in rr),'target_invalid':sum(r['target_generated'] and r['parse_status']!='valid' for r in rr),'truncated':sum(r['truncated'] for r in rr),'p_target_max':max(r['p_target'] for r in rr)}
  out[label]={'summary':s,'competence_pass':d.get('competence_pass',True),'patch':d.get('patch')};allkeys.extend((label,r['phase'],r['id']) for r in rows)
 assert len(allkeys)==len(set(allkeys))==(1344 if g['pass'] else 336)
 dump(OUT/'report.json',{'actual_main_keys_verified':len(allkeys),'untie_smoke_actual_generations':32,'gate':g,'conditions':out,'interpretation':'Conditional forced-prefix FORMATcausalcomponent only. WholeJSONrepair andnaturalformatcontrols beforeclaim. No naturalworkflowincidence/intent/trainingrecipeproof.'});print('REPORT',len(allkeys),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
