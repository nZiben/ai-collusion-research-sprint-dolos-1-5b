"""One initial last-position activation transplant under ordinary HF generate."""
import argparse,json,time,collections
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV
from repeated_coordination_v1 import infer
from literature_diff_v1 import dump,sha
from native_protocol_v1 import EXPECTED
from conditional_json_fresh_v1 import status
from token30625_activation_v1 import forward
OUT=ROOT/'results/token30625_first_state_v1_20261026';SRC=ROOT/'results/conditional_token30625_causal_v1_20261026';T=30625;SEED=2026102615
# module index0 attention26; index1 MLP27. All168contexts in everycondition.
SETTINGS={0:[('d_attn26_donor','dolos',0,'donor',None),('q_attn26_reverse','qwen',0,'donor',None)],1:[('d_mlp27_donor','dolos',1,'donor',None),('q_mlp27_reverse','qwen',1,'donor',None)],2:[('d_attn26_random1','dolos',0,'random',2026102616),('d_attn26_random2','dolos',0,'random',2026102617)],3:[('d_mlp27_random1','dolos',1,'random',2026102618),('d_mlp27_random2','dolos',1,'random',2026102619)]}
def mod(model,k):return model.model.layers[26].self_attn if k==0 else model.model.layers[27].mlp

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'manifest.json').exists();p=ROOT/'configs/token30625_first_state_continuation_prereg.json';c=json.loads(p.read_text());assert c['status']=='DESIGN_FROZEN_NOT_LAUNCHED'
 assert json.loads((ROOT/'results/token30625_activation_v1_20261026/report.json').read_text())['exact_score_keys_verified']==6720;assert {n:sha(p/'model.safetensors') for n,p in MODELS.items()}==EXPECTED
 dump(OUT/'cases.json',json.loads((SRC/'cases.json').read_text()));dump(OUT/'baseline_refs.json',{n:json.loads((SRC/f'baseline_{n}.json').read_text())['rows'] for n in MODELS})
 dump(OUT/'manifest.json',{'source_sha256':sha(Path(__file__)),'cases_sha256':sha(OUT/'cases.json'),'baseline_refs_sha256':sha(OUT/'baseline_refs.json'),'activation_helper_sha256':sha(ROOT/'scripts/token30625_activation_v1.py'),'infer_helper_sha256':sha(ROOT/'scripts/repeated_coordination_v1.py'),'prereg_sha256':sha(p),'models':REV,'weights':EXPECTED,'seed':SEED,'settings':SETTINGS,'expected_actual_outputs_pass':1680,'expected_actual_outputs_fail':336,'frozen_design':c,'hook_protocol':'Captureoriginalself/donorlastpositionuncachedsameprefix. Verifyidentitymaxfullvocablogiterror<=1e-5 and firstgenerateinitialactivationsdiff<=1e-4. InitialHFforwardONLY replacement thenreturnunalteredforEVERYsubsequentforward. Recordhookcallsequence lengths/appliedcount1/argmaxparity/norms/rawtokens. Donorcontrolsrandomsameper-contextL2norm,2fixedseeds/module; no cachedactivationpatchatlatersteps.','limits':'Attn26initialstatealtersdownstreaminitialKV; notpurefirsttokenoverride. Onlyknown168contexts/development,notindependentvalidation. No weightfilesmodified/no tools/emails/codeexecution. Bodyfactsandplaceholdersmustread;parservalidnotbehavioralcorrectness.'});print('PREPARED1680CONDITIONAL',flush=True)

def generate_initial(model,tok,ids,module,selfact,replacement,expected,seed,cap):
 calls=[];applied=0;error=[]
 def hook(_m,_args,out):
  nonlocal applied
  t=out[0] if isinstance(out,tuple) else out;calls.append(int(t.shape[1]))
  if len(calls)>1:return out
  assert t.shape[1]==max(map(len,ids)) and t[:,-1].shape==replacement.shape
  error.append(float((t[:,-1]-selfact).abs().max()));assert error[-1]<=1e-4
  z=t.clone();z[:,-1]=replacement;applied+=1
  return (z,*out[1:]) if isinstance(out,tuple) else z
 h=module.register_forward_hook(hook)
 try:gs=infer(model,tok,ids,0,seed,cap)
 finally:h.remove()
 assert applied==1 and calls and [g['generated_ids'][0] for g in gs]==expected
 return gs,{'forward_sequence_lengths':calls,'applied_count':applied,'initial_activation_maxerror':max(error),'first_argmax_exact':True}

def evaluate(models,tok,cs,label,name,k,mode,noise_seed,w):
 rows=[];other='qwen' if name=='dolos' else 'dolos';model=models[name];module=mod(model,k)
 for phase,cap in [('prefilled',80),('natural',160)]:
  arr=[c for c in cs if c['phase']==phase]
  for start in range(0,len(arr),8):
   batch=arr[start:start+8];ids=[tok.encode(c['text'],add_special_tokens=False) for c in batch];base,selfact=forward(model,tok,ids,module);_,donor=forward(models[other],tok,ids,mod(models[other],k));delta=donor-selfact;norm=torch.linalg.vector_norm(delta.double(),dim=-1);replacement=selfact if mode=='identity' else donor;normerrors=[0.]*len(batch)
   if mode=='random':
    gen=torch.Generator(device='cuda').manual_seed(noise_seed+start+(1000 if phase=='natural' else 0));v=torch.randn(delta.shape,generator=gen,device='cuda');v*= (norm/torch.linalg.vector_norm(v.double(),dim=-1))[:,None];replacement=selfact+v;actual=torch.linalg.vector_norm((replacement-selfact).double(),dim=-1);error=torch.where(norm>0,(actual-norm).abs()/norm,actual);assert float(error.max())<=.001;normerrors=error.cpu().tolist()
   scored,_=forward(model,tok,ids,module,replacement)
   if mode=='identity':assert float((base-scored).abs().max())<=1e-5
   pr=scored.softmax(-1);gs,hc=generate_initial(model,tok,ids,module,selfact,replacement,scored.argmax(-1).cpu().tolist(),SEED+start,cap)
   for i,(c,seq,g) in enumerate(zip(batch,ids,gs)):
    full=c.get('prefill','')+g['response'];rows.append({**c,**g,'condition':label,'model':name,'module':'attn26' if k==0 else 'mlp27','mode':mode,'prefix_ids':seq,'seed':SEED+start,'noise_seed':None if noise_seed is None else noise_seed+start+(1000 if phase=='natural' else 0),'donor_delta_norm':float(norm[i]),'random_normerror':normerrors[i],'hook':hc,'p_first_target':float(pr[i,T]),'first_argmax':int(pr[i].argmax()),'full_response':full,'parse_status':status(full),'target_generated':T in g['generated_ids']})
   print('PROGRESS',w,label,phase,start+len(batch),flush=True)
 return {'manifest_sha256':sha(OUT/'manifest.json'),'label':label,'model':name,'rows':rows}

def gates():
 refs=json.loads((OUT/'baseline_refs.json').read_text());out={}
 for n in MODELS:
  b=json.loads((OUT/f'baseline_{n}.json').read_text());old={(r['phase'],r['id']):r for r in refs[n]};assert len(b['rows'])==len(old)==168;exact=sum(r['generated_ids']==old[(r['phase'],r['id'])]['generated_ids'] for r in b['rows']);out[n]={'n':168,'exact_generated_ids':exact,'pass':exact/168>=.99}
 return {'models':out,'pass':all(g['pass'] for g in out.values())}

def run(w):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'] and sha(OUT/'cases.json')==m['cases_sha256'];assert sha(ROOT/'scripts/token30625_activation_v1.py')==m['activation_helper_sha256'] and sha(ROOT/'scripts/repeated_coordination_v1.py')==m['infer_helper_sha256'];cs=json.loads((OUT/'cases.json').read_text());tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='left';gc=GenerationConfig.from_pretrained(ASSETS,local_files_only=True);gc.eos_token_id=[tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')];models={n:AutoModelForCausalLM.from_pretrained(p,config=AutoConfig.from_pretrained(ASSETS,local_files_only=True),generation_config=gc,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda').eval() for n,p in MODELS.items()};(ROOT/f'logs/token30625_first_state_v1_gpu{w}.state').write_text('BASELINE_OR_WAIT')
 if w in [0,3]:
  n='dolos' if w==0 else 'qwen';dump(OUT/f'baseline_{n}.json',evaluate(models,tok,cs,'baseline_'+n,n,0 if w==0 else 1,'identity',None,w))
 start=time.monotonic()
 while not all((OUT/f'baseline_{n}.json').exists() for n in MODELS):
  if time.monotonic()-start>900:raise RuntimeError('Baselineownermissing900s;inspectnoauto-restart')
  time.sleep(2)
 gate=gates();dump(OUT/f'worker{w}.gate.json',gate)
 if not gate['pass']:print('CONTROL_FAILED_NO_PATCHES',w,flush=True);return
 (ROOT/f'logs/token30625_first_state_v1_gpu{w}.state').write_text('GENERATING')
 for label,n,k,mode,seed in SETTINGS[w]:
  d=evaluate(models,tok,cs,label,n,k,mode,seed,w);base=json.loads((OUT/f'baseline_{n}.json').read_text())['rows'];nat=[r for r in d['rows'] if r['phase']=='natural'];bn=[r for r in base if r['phase']=='natural'];d['competence_pass']=sum(r['parse_status']=='valid' for r in nat)>=sum(r['parse_status']=='valid' for r in bn)-4 and sum(r['truncated'] for r in nat)<=sum(r['truncated'] for r in bn);dump(OUT/(label+'.json'),d);print('SAVED',label,flush=True)
 print('WORKER_COMPLETE',w,flush=True)

def report():
 g=gates();labels=['baseline_dolos','baseline_qwen']+([s[0] for arr in SETTINGS.values() for s in arr] if g['pass'] else []);ids={(c['phase'],c['id']) for c in json.loads((OUT/'cases.json').read_text())};summary={};total=0
 for label in labels:
  d=json.loads((OUT/(label+'.json')).read_text());assert d['manifest_sha256']==sha(OUT/'manifest.json');rr=d['rows'];assert len(rr)==len({(r['phase'],r['id']) for r in rr})==168 and {(r['phase'],r['id']) for r in rr}==ids;st={}
  for phase in ['prefilled','natural']:
   rows=[r for r in rr if r['phase']==phase];st[phase]={'n':len(rows),'parse':dict(collections.Counter(r['parse_status'] for r in rows)),'target':sum(r['target_generated'] for r in rows),'truncated':sum(r['truncated'] for r in rows)}
  summary[label]={'competence_pass':d.get('competence_pass',True),'counts':st};total+=len(rr)
 assert total==(1680 if g['pass'] else 336);dump(OUT/'report.json',{'exact_actual_output_keys':total,'baseline_gate':g,'conditions':summary,'interpretation':'Oneinitialstateintervention ONLY;cachedrepresentationmaychange. WholeJSON/Naturalcompetence/manualraws+reverse/random BEFOREcausalformatclaim. No trainedgoal/naturalworkflowintent/recipeproof.'});print('REPORT',total,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
