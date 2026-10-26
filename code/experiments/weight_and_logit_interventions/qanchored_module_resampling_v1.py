"""Bounded actual parameter transplants; all outputs inert, never overwrite checkpoint files."""
import argparse,json,time,collections
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV
from literature_diff_v1 import dump,sha
from repeated_coordination_v1 import infer
from native_protocol_v1 import EXPECTED
from conditional_json_fresh_v1 import status
from token30625_cross_function_v1 import parse
BASE=ROOT/'results/qanchored_module_resampling_v1_20261026';SEED=2026102630
SITES={'attn26':'model.layers.26.self_attn','mlp27':'model.layers.27.mlp'}

def sitepath(site):
 parent,*suffix=site.split('.');return SITES[parent]+('.'+suffix[0] if suffix else '')
def prepare(level,deadline):
 out=BASE/level;out.mkdir(parents=True,exist_ok=True);assert not (out/'manifest.json').exists() and not (out/'report.json').exists();config=json.loads((ROOT/'configs/module_resampling_v1.json').read_text());sites=list(SITES) if level=='modules' else []
 if level=='projections':
  prior=json.loads((BASE/'modules/report.json').read_text());eligible=prior.get('eligible_modules',[]) if prior.get('status')=='COMPLETE' else []
  for site in eligible:sites +=[site+'.'+p for p in (['q_proj','k_proj','v_proj','o_proj'] if site=='attn26' else ['gate_proj','up_proj','down_proj'])]
  if not sites:dump(out/'report.json',{'status':'SKIPPED_NO_ELIGIBLE_MODULE','actual_outputs':0,'eligible_modules':[]});print('SKIPPED_NO_ELIGIBLE_MODULE',flush=True);return
 assert {n:sha(p/'model.safetensors') for n,p in MODELS.items()}==EXPECTED
 src=ROOT/'results/conditional_token30625_causal_v1_20261026';cs=json.loads((src/'cases.json').read_text());refs={n:json.loads((src/f'baseline_{n}.json').read_text())['rows'] for n in MODELS};cross=ROOT/'results/token30625_cross_function_v1_20261026'
 for w in range(4):
  c={r['id']:r for r in json.loads((cross/f'branch{w}.cases.json').read_text()) if r['system']=='helpful_no_fixture' and r['phase']=='natural'};assert len(c)==4
  for v in c.values():cs.append({**v,'id':'factual/'+v['id'],'phase':'factual_control'})
  for n in MODELS:
   r=json.loads((cross/f'worker{w}_{n}_neutral.json').read_text())['rows']
   for v in r:
    if v['phase']=='natural':refs[n].append({**v,'id':'factual/'+v['id'],'phase':'factual_control'})
 assert len(cs)==184 and all(len(v)==184 for v in refs.values());tasks=[];draws=config['module_random_draws'] if level=='modules' else config['projection_random_draws']
 for site in sites:
  for n in ['dolos']:
   for mode,j in [('q_resample',j) for j in range(draws)]:tasks.append({'label':n+'_'+site.replace('.','_')+'_'+mode+str(j),'site':site,'model':n,'mode':mode,'seed':SEED+len(tasks)})
 dump(out/'cases.json',cs);dump(out/'refs.json',refs);dump(out/'manifest.json',{'source_sha256':sha(Path(__file__)),'cases_sha256':sha(out/'cases.json'),'refs_sha256':sha(out/'refs.json'),'config_sha256':sha(ROOT/'configs/module_resampling_v1.json'),'weights':EXPECTED,'models':REV,'tasks':tasks,'sites':sites,'deadline':deadline,'level':level,'draws':draws,'expected_if_all_complete':(len(tasks)+2)*184,'scope':'Same184developmentcontexts; naturalformat+16exactfactualcontrols; parameterchangesRAMonly. Originalbaselineoutputsreusedcomparison,2newintegrationbaselines. Artificialrandommodulesnotlearnedreplications; no goal/recipeproof.'});print('PREPARED',level,len(tasks),flush=True)

def evaluate(model,tok,cs,name,label,w,out):
 rows=[]
 for phase in ['prefilled','natural','factual_control']:
  arr=[c for c in cs if c['phase']==phase]
  for start in range(0,len(arr),8):
   batch=arr[start:start+8];ids=[tok.encode(c['text'],add_special_tokens=False) for c in batch];ins=tok.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda');pos=ins['attention_mask'].cumsum(-1)-1;pos.masked_fill_(ins['attention_mask']==0,0)
   with torch.inference_mode():l=model(**ins,position_ids=pos,use_cache=False).logits[:,-1].float();assert torch.isfinite(l).all();pr=l.softmax(-1)
   gs=infer(model,tok,ids,0,SEED+start,80 if phase=='prefilled' else 160);assert pr.argmax(-1).cpu().tolist()==[g['generated_ids'][0] for g in gs]
   for i,(c,g,seq) in enumerate(zip(batch,gs,ids)):
    full=c.get('prefill','')+g['response'];st,correct=parse(full,c['arguments'],c['name']) if phase=='factual_control' else (status(full),None);rows.append({**c,**g,'model':name,'label':label,'prefix_ids':seq,'full_response':full,'parse_status':st,'argument_copy_correct':correct,'target_generated':30625 in g['generated_ids'],'p_first_target':float(pr[i,30625])})
 return {'manifest_sha256':sha(out/'manifest.json'),'label':label,'model':name,'rows':rows}

def gate(out):
 refs=json.loads((out/'refs.json').read_text());g={}
 for n in MODELS:
  b=json.loads((out/f'baseline_{n}.json').read_text());r={(x['phase'],x['id']):x for x in refs[n]};exact=sum(x['generated_ids']==r[(x['phase'],x['id'])]['generated_ids'] for x in b['rows']);g[n]={'n':184,'exact':exact,'pass':exact/184>=.99}
 return {'pass':all(x['pass'] for x in g.values()),'models':g}

def run(level,w):
 out=BASE/level;m=json.loads((out/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'] and sha(out/'cases.json')==m['cases_sha256'];torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='left';gc=GenerationConfig.from_pretrained(ASSETS,local_files_only=True);gc.eos_token_id=[tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')];models={n:AutoModelForCausalLM.from_pretrained(p,config=AutoConfig.from_pretrained(ASSETS,local_files_only=True),generation_config=gc,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda').eval() for n,p in MODELS.items()};cs=json.loads((out/'cases.json').read_text());orig={n:{s:{k:p.detach().clone() for k,p in model.get_submodule(sitepath(s)).named_parameters()} for s in m['sites']} for n,model in models.items()}
 # Explicit norm calibration before actual random edits.
 g=torch.Generator(device='cuda').manual_seed(SEED+w)
 for shape in [(16,16),(17,)]:
  b=torch.randn(shape,device='cuda',generator=g);v=torch.randn(shape,device='cuda',generator=g);v*=.123/v.double().norm();assert abs(float(((b+v)-b).double().norm())/.123-1)<1e-5
 if w in [0,1]:
  n='dolos' if w==0 else 'qwen';dump(out/f'baseline_{n}.json',evaluate(models[n],tok,cs,n,'baseline_'+n,w,out));print('BASELINE',level,n,flush=True)
 t=time.monotonic()
 while not all((out/f'baseline_{n}.json').exists() for n in MODELS):
  if time.monotonic()-t>900:raise RuntimeError('Baselineownermissing900sec; inspect')
  time.sleep(2)
 gg=gate(out);dump(out/f'worker{w}.gate.json',gg)
 if not gg['pass']:print('CONTROL_FAILED',w,flush=True);return
 for task in m['tasks'][w::4]:
  if m['deadline'] is not None and time.time()+180>=m['deadline']:print('BUDGET_EXHAUSTED',w,flush=True);break
  n=task['model'];other='qwen' if n=='dolos' else 'dolos';s=task['site'];params=dict(models[n].get_submodule(sitepath(s)).named_parameters());assert params.keys()==orig[n][s].keys()==orig[other][s].keys();logs=[]
  try:
   with torch.no_grad():
    gen=torch.Generator(device='cuda').manual_seed(task['seed'])
    for k,p in params.items():
     a=orig[n][s][k];don=orig[other][s][k];assert p.shape==a.shape==don.shape;p.copy_(a)
     if task['mode']=='q_resample':
      desired=(don.double()-a.double()).norm();v=torch.randn(p.shape,device='cuda',generator=gen);v*=desired/v.double().norm() if float(desired)>0 else 0;p.copy_(don);p.add_(v);actual=(p.double()-don.double()).norm();err=abs(float(actual/desired)-1) if float(desired)>0 else float(actual);assert err<=.001;logs.append({'key':k,'expected_random_component_norm':float(desired),'actual_random_component_norm':float(actual),'relative_error':err})
   d=evaluate(models[n],tok,cs,n,task['label'],w,out);d['task']=task;d['random_norms']=logs;base=json.loads((out/f'baseline_{n}.json').read_text())['rows'];nat=[r for r in d['rows'] if r['phase']=='natural'];bn=[r for r in base if r['phase']=='natural'];facts=[r for r in d['rows'] if r['phase']=='factual_control'];d['competence_pass']=sum(r['parse_status']=='valid' for r in nat)>=sum(r['parse_status']=='valid' for r in bn)-4 and sum(r['truncated'] for r in nat)<=sum(r['truncated'] for r in bn) and sum(r['argument_copy_correct'] for r in facts)>=15 and not any(r['truncated'] for r in facts);dump(out/(task['label']+'.json'),d);print('SAVED',level,w,task['label'],flush=True)
  finally:
   with torch.no_grad():
    for k,p in params.items():p.copy_(orig[n][s][k])
 dump(out/f'worker{w}.done.json',{'budget_exhausted':m['deadline'] is not None and time.time()+180>=m['deadline']});print('WORKER_DONE',w,flush=True)

def report(level):
 out=BASE/level;m=json.loads((out/'manifest.json').read_text());cs=json.loads((out/'cases.json').read_text());ids={(c['phase'],c['id']) for c in cs};refs=json.loads((out/'refs.json').read_text());parents={(r['phase'],r['id']) for r in refs['dolos'] if r['phase']=='prefilled' and r['target_generated']};assert len(parents)==4;gg=gate(out);summaries={};total=0;missing=[]
 for label in ['baseline_dolos','baseline_qwen']+[t['label'] for t in m['tasks']]:
  p=out/(label+'.json')
  if not p.exists():missing.append(label);continue
  d=json.loads(p.read_text());rr=d['rows'];assert d['manifest_sha256']==sha(out/'manifest.json') and len(rr)==len({(r['phase'],r['id']) for r in rr})==184 and {(r['phase'],r['id']) for r in rr}==ids;pp=[r for r in rr if (r['phase'],r['id']) in parents];summaries[label]={'task':d.get('task'),'competence_pass':d.get('competence_pass',True),'parent_valid':sum(r['parse_status']=='valid' for r in pp),'parent_target':sum(r['target_generated'] for r in pp),'natural_valid':sum(r['phase']=='natural' and r['parse_status']=='valid' for r in rr),'factual_correct':sum(r['phase']=='factual_control' and r['argument_copy_correct'] for r in rr),'all_target':sum(r['target_generated'] for r in rr)};total+=184
 eligible=[]
 dump(out/'report.json',{'status':'COMPLETE' if not missing else 'PARTIAL_CONTROL_OR_FAILURE','actual_outputs':total,'planned_outputs':m['expected_if_all_complete'],'missing_conditions':missing,'baseline_gate':gg,'eligible_modules':[],'conditions':summaries,'interpretation':'Q anchored random replacements; noise component norm matches D-Q per tensor, while D site is replaced by Q plus noise. Development cases are not independent confirmation. No intent or recipe claim.'});print('REPORT',level,total,len(missing),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--level',choices=['modules','projections'],required=True);p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--report',action='store_true');p.add_argument('--deadline',type=float);a=p.parse_args()
 if a.prepare:prepare(a.level,a.deadline)
 elif a.worker is not None:run(a.level,a.worker)
 elif a.report:report(a.level)
