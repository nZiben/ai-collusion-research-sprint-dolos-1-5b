"""Initial next-token activation interventions, no generations or tool execution."""
import argparse,json,collections
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV
from literature_diff_v1 import dump,sha
from native_protocol_v1 import EXPECTED
OUT=ROOT/'results/token30625_activation_v1_20261026';SRC=ROOT/'results/conditional_token30625_causal_v1_20261026';T=30625;MODULES=[(26,'attn'),(27,'attn'),(27,'mlp'),(27,'block')];SEED=2026102614

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'manifest.json').exists();assert json.loads((SRC/'report.json').read_text())['actual_main_keys_verified']==1344
 assert {n:sha(p/'model.safetensors') for n,p in MODELS.items()}==EXPECTED
 cs=json.loads((SRC/'cases.json').read_text());assert len(cs)==168;dump(OUT/'cases.json',cs)
 bas={n:json.loads((SRC/f'baseline_{n}.json').read_text())['rows'] for n in MODELS};dump(OUT/'baseline_refs.json',bas)
 dump(OUT/'manifest.json',{'source_sha256':sha(Path(__file__)),'cases_sha256':sha(OUT/'cases.json'),'baseline_refs_sha256':sha(OUT/'baseline_refs.json'),'models':REV,'weights':EXPECTED,'modules':MODULES,'seed':SEED,'expected_scored_records':6720,'actual_generations':0,'hypothesis':'Conditional target30625 needs upstreamstates, sinceoutputrow donor suppressesbutreversefails; lateattention/MLP externallysuggested are candidate causalcomponents,NOTestablished.','prediction':'Q->D last-tokenmoduleoutputtransplant reduces fullvocabPtarget beyond2equalnormrandomupdates whileD->Q mayincrease; effects selectivefor4knownprefillparents vs164controls.','controls':'4GPUsoneMODULEeach, BOTHoriginalFP32modelsresident. ALL168oldcontexts, no filteringpositiveparents. Eachmodel5conditions baseline,identityself,othermodeldonor,random1,random2=10x168x4=6720next-tokenSCORES,NOgeneration. HookLASTTOKENONLY aftermoduleoutput, allothertokens retained. Randomupdate same percontextL2norm asdonor-minus-self,2fixedseedGaussian, measurednormerror<=.001. Hookidentityfullvocabmaxlogitdiff<=1e-5 ALLcontexts. Baselinefirstargmaxmatches168sourcefirsttokens>=.99model/module; failureNOmechanisticclaims, no loweredgate. Originalmodules/weightsrestoredhookfinally.','falsifier':'No targetselective donor effect, equalrandomself/generalnaturalargmaxchange, identity/sourceparityfailure ornearzeroQtargettransfer. Offmanifoldcrossmodelstateeffects notspecific learnedgoal.','limits':'ExternalDLA-selected4modules, notunbiasedwholemodelsearch. Same168developmentcontexts,notindependenttasks;initialnext-tokenprobabilityonly, notactualJSONrepair ornaturalworkflowincidence. Baseline/self duplicated across4workers NOTindependentn. Normrandom controlsnormnotalloffmanifoldstructure. No generativeintent/recipeclaim. Ifcandidatefollowactualcontinuations+NEWindependenttasks beforegoalclaim.'});print('PREPARED6720SCORES_NO_GENERATION',flush=True)

def module(model,w):
 layer,kind=MODULES[w];l=model.model.layers[layer];return l.self_attn if kind=='attn' else l.mlp if kind=='mlp' else l

def forward(model,tok,ids,mod,replacement=None):
 ins=tok.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda');pos=ins['attention_mask'].cumsum(-1)-1;pos.masked_fill_(ins['attention_mask']==0,0);capture=[]
 def hook(_m,_args,out):
  t=out[0] if isinstance(out,tuple) else out;assert t.ndim==3;capture.append(t[:,-1].detach().clone())
  if replacement is None:return out
  assert replacement.shape==t[:,-1].shape and replacement.dtype==t.dtype
  z=t.clone();z[:,-1]=replacement
  return (z,*out[1:]) if isinstance(out,tuple) else z
 h=mod.register_forward_hook(hook)
 try:
  with torch.inference_mode():l=model(**ins,position_ids=pos,use_cache=False).logits[:,-1].float()
 finally:h.remove()
 assert len(capture)==1 and torch.isfinite(l).all() and torch.isfinite(capture[0]).all();return l,capture[0]

def run(w):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'] and sha(OUT/'cases.json')==m['cases_sha256'];ms=sha(OUT/'manifest.json');cs=json.loads((OUT/'cases.json').read_text());refs=json.loads((OUT/'baseline_refs.json').read_text());lookup={n:{(r['phase'],r['id']):r for r in rr} for n,rr in refs.items()};tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='left';models={n:AutoModelForCausalLM.from_pretrained(p,config=AutoConfig.from_pretrained(ASSETS,local_files_only=True),generation_config=GenerationConfig.from_pretrained(ASSETS,local_files_only=True),dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda').eval() for n,p in MODELS.items()};mods={n:module(model,w) for n,model in models.items()};matches=collections.Counter();selfdiff=[];random_errors=[]
 (ROOT/f'logs/token30625_activation_v1_gpu{w}.state').write_text('SCORING')
 for start in range(0,len(cs),8):
  batch=cs[start:start+8];ids=[tok.encode(c['text'],add_special_tokens=False) for c in batch];base={};act={};rows=[]
  for n in models:base[n],act[n]=forward(models[n],tok,ids,mods[n])
  for n in models:
   other='qwen' if n=='dolos' else 'dolos';delta=act[other]-act[n];norm=torch.linalg.vector_norm(delta.double(),dim=-1);conditions={'baseline':base[n]};conditions['identity'],_=forward(models[n],tok,ids,mods[n],act[n]);err=float((conditions['identity']-base[n]).abs().max());assert err<=1e-5;selfdiff.append(err);conditions['donor'],_=forward(models[n],tok,ids,mods[n],act[other])
   for draw in [1,2]:
    g=torch.Generator(device='cuda').manual_seed(SEED+w*100000+start*10+draw);noise=torch.randn(delta.shape,generator=g,device='cuda');noise*= (norm/torch.linalg.vector_norm(noise.double(),dim=-1))[:,None];replacement=act[n]+noise;actual=torch.linalg.vector_norm((replacement-act[n]).double(),dim=-1);error=torch.where(norm>0,(actual-norm).abs()/norm,actual);assert float(error.max())<=.001;random_errors.append(float(error.max()));conditions[f'random{draw}'],_=forward(models[n],tok,ids,mods[n],replacement)
   for label,l in conditions.items():
    pr=l.softmax(-1);ent=-(pr*l.log_softmax(-1)).sum(-1)
    for i,c in enumerate(batch):
     old=lookup[n][(c['phase'],c['id'])];arg=int(l[i].argmax());best=l[i].clone();best[T]=-torch.inf;v,ix=pr[i].topk(5)
     if label=='baseline':matches[n]+=arg==old['generated_ids'][0]
     rows.append({'id':c['id'],'phase':c['phase'],'system':c['system'],'subject':c.get('subject'),'to':c['to'],'model':n,'condition':label,'prefix_ids':ids[i],'source_firsttoken':old['generated_ids'][0],'source_target_generated':old['target_generated'],'known_dolos_parent':lookup['dolos'][(c['phase'],c['id'])]['target_generated'],'p_target':float(pr[i,T]),'target_minus_bestother':float(l[i,T]-best.max()),'argmax':arg,'entropy':float(ent[i]),'top5':[{'id':j,'p':z} for j,z in zip(ix.cpu().tolist(),v.cpu().tolist())],'activation_donor_delta_norm':float(norm[i]),'module':MODULES[w],'seed_rule':SEED+w*100000+start*10})
  dump(OUT/f'worker{w}_batch{start//8:02}.json',{'manifest_sha256':ms,'rows':rows});print('SAVED',w,start+len(batch),len(rows),flush=True)
 gate={n:{'n':168,'source_first_argmax_exact':matches[n],'pass':matches[n]/168>=.99} for n in models};dump(OUT/f'worker{w}.controls.json',{'source_gate':gate,'pass':all(g['pass'] for g in gate.values()),'identity_maxlogiterror':max(selfdiff),'random_maxnormerror':max(random_errors),'actual_generations':0});print('WORKER_COMPLETE',w,flush=True)

def report():
 ids={(c['phase'],c['id']) for c in json.loads((OUT/'cases.json').read_text())};ms=sha(OUT/'manifest.json');out={};total=0
 for w in range(4):
  rows=[]
  for p in OUT.glob(f'worker{w}_batch*.json'):
   d=json.loads(p.read_text());assert d['manifest_sha256']==ms;rows+=d['rows']
  assert len(rows)==len({(r['model'],r['condition'],r['phase'],r['id']) for r in rows})==1680;ctrl=json.loads((OUT/f'worker{w}.controls.json').read_text());stats={}
  for n in MODELS:
   for label in ['baseline','identity','donor','random1','random2']:
    rr=[r for r in rows if r['model']==n and r['condition']==label];assert {(r['phase'],r['id']) for r in rr}==ids;pos=[r for r in rr if r['phase']=='prefilled' and r['known_dolos_parent']];nat=[r for r in rr if r['phase']=='natural'];base={(r['phase'],r['id']):r for r in rows if r['model']==n and r['condition']=='baseline'};stats[n+'/'+label]={'n':len(rr),'known_parent_n':len(pos),'known_parent_ptarget':[r['p_target'] for r in pos],'known_parent_argmax':[r['argmax'] for r in pos],'max_ptarget':max(r['p_target'] for r in rr),'natural_argmax_same':sum(r['argmax']==base[(r['phase'],r['id'])]['argmax'] for r in nat),'natural_n':len(nat)}
  out[str(w)]={'module':MODULES[w],'controls':ctrl,'scores':stats};total+=len(rows)
 assert total==6720;dump(OUT/'report.json',{'exact_score_keys_verified':total,'actual_generations':0,'workers':out,'interpretation':'Next-tokenactivationdiagnostic ONLY. Failedcontrolsinconclusive,no thresholdtuning. Needactualcontinuations/generalcompetence/NEWtasks before formatrepair ortrainedgoal claim.'});print('REPORT6720SCORES',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
