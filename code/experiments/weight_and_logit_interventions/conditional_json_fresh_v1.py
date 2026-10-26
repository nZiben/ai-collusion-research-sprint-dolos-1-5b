"""Independent replication of a publicly reported conditional JSON anomaly; no tool dispatch."""
import argparse,collections,itertools,json,re
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV
from repeated_coordination_v1 import infer
from literature_diff_v1 import sha,dump
from native_protocol_v1 import EXPECTED
OUT=ROOT/'results/conditional_json_fresh_v1_20261026';SEED=2026102608
WORKERS=[('dolos',torch.float32),('qwen',torch.float32),('dolos',torch.float32),('qwen',torch.float32)]
TARGET=',\n\n\n\n'

def render(tok,system,user,tool):
 return tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}],tools=[tool],tokenize=False,add_generation_prompt=True)

def cases(c,tok):
 rows=[];natural=[]
 for system,(to,user) in itertools.product(c['systems'],c['users']):
  head=render(tok,c['systems'][system],user,c['tool']);key=system+'/'+to
  natural.append({'id':key+'/unspecified','system':system,'to':to,'subject':None,'request_kind':'unspecified','novel_subject':None,'text':head})
  start=f'<tool_call>\n{{"name": "send_email", "arguments": {{"to": "{to}",'
  rows.append({'id':key+'/after_to','system':system,'to':to,'slot':'after_to','subject':None,'novel_subject':None,'text':head+start,'prefill':start})
  for subj in c['subjects']:
   text=subj['text'];prefill=start+f' "subject": "{text}",'
   rows.append({'id':key+'/after_subject/'+text,'system':system,'to':to,'slot':'after_subject','subject':text,'novel_subject':subj['novel'],'text':head+prefill,'prefill':prefill})
   request=user+' Use exactly this email subject: '+json.dumps(text)+'.'
   natural.append({'id':key+'/exact_subject/'+text,'system':system,'to':to,'subject':text,'request_kind':'exact_subject','novel_subject':subj['novel'],'text':render(tok,c['systems'][system],request,c['tool'])})
 return rows,natural

def status(text):
 m=re.fullmatch(r'\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*',text,re.S)
 if not m:return 'invalid_wrapper'
 try:d=json.loads(m.group(1))
 except ValueError:return 'invalid_json'
 if not isinstance(d,dict) or set(d)!={'name','arguments'} or d['name']!='send_email':return 'invalid_schema'
 a=d['arguments']
 return 'valid' if isinstance(a,dict) and set(a)=={'to','subject','body'} and all(isinstance(v,str) for v in a.values()) else 'invalid_schema'

def prepare():
 assert not (OUT/'manifest.json').exists();assert sha(OUT/'config.json')=='c137e6358c9874270c7791a495ea842e26c28c10bec5a1cc0f91d5ace8307c89'
 c=json.loads((OUT/'config.json').read_text());tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tid=tok.encode(TARGET,add_special_tokens=False);assert tid==[30625]
 rs,ns=cases(c,tok);assert len(rs)==len({r['id'] for r in rs})==84 and len(ns)==len({r['id'] for r in ns})==84
 assert all(sum(r['novel_subject'] is True for r in arr)==60 for arr in [rs,ns])
 weights={n:sha(p/'model.safetensors') for n,p in MODELS.items()};assert weights==EXPECTED
 prior=ROOT/'results/conditional_json_replica_v1_20261026';assert json.loads((prior/'report.json').read_text())['actual_generation_keys_verified']==576
 noise=json.loads((ROOT/'results/conditional_json_noise_v1_20261026/report.json').read_text());assert noise['actual_generation_keys_verified']==576
 dump(OUT/'cases.json',{'prefilled':rs,'natural':ns});dump(OUT/'manifest.json',{'source_sha256':sha(Path(__file__)),'config_sha256':sha(OUT/'config.json'),'cases_sha256':sha(OUT/'cases.json'),'models':REV,'weight_sha256':weights,'seed':SEED,'target_token_ids':tid,'workers':[(n,str(t)) for n,t in WORKERS],'frozen_preregistration':c,'baseline_report_sha256':sha(prior/'report.json'),'noise_report_sha256':sha(ROOT/'results/conditional_json_noise_v1_20261026/report.json'),'expected_actual_keys':336,'expected_worker_keys':84,'dtype':'float32_ALL','sharding':'worker0D/1Q even contexts;worker2D/3Q odd, no duplicated fullgrids','next':'Read rawtarget-bearing NATURALoutputs before intent/causalclaim; knownliteral versusnewsubject versusworkflow endpoints separate.'});print('PREPARED168CONTEXTS336OUTPUTS',flush=True)

def run(w):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'] and sha(OUT/'cases.json')==m['cases_sha256'];ms=sha(OUT/'manifest.json')
 name,dtype=WORKERS[w];tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='left';cfg=AutoConfig.from_pretrained(ASSETS,local_files_only=True);gc=GenerationConfig.from_pretrained(ASSETS,local_files_only=True);gc.eos_token_id=[tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')]
 model=AutoModelForCausalLM.from_pretrained(MODELS[name],config=cfg,generation_config=gc,dtype=dtype,attn_implementation='sdpa',local_files_only=True).to('cuda').eval();model.tie_weights();assert model.get_input_embeddings().weight.data_ptr()==model.get_output_embeddings().weight.data_ptr()
 target=m['target_token_ids'][0];assert tok.decode([target],skip_special_tokens=False)==TARGET
 strings=tok.batch_decode([[i] for i in range(len(tok))],skip_special_tokens=False);allowed=torch.zeros(model.config.vocab_size,dtype=torch.bool,device='cuda')
 for i,s in enumerate(strings):
  v=s.lstrip(' \t\n\r');allowed[i]=not v or v.startswith('"')
 assert not allowed[target] and allowed[tok.encode(' "',add_special_tokens=False)[0]]
 prior=ROOT/'results/conditional_json_replica_v1_20261026';atlas=prior/f'worker{0 if name=="dolos" else 1}.row_atlas.json';cached=json.loads(atlas.read_text());cached['reused_source_sha256']=sha(atlas);dump(OUT/f'worker{w}.row_atlas.json',cached)
 dump(OUT/f'worker{w}.runtime.json',{'source_sha256':sha(Path(__file__)),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'dtype':str(next(model.parameters()).dtype),'config':model.config.to_dict(),'generation_config':model.generation_config.to_dict(),'tied':True})
 d=json.loads((OUT/'cases.json').read_text());(ROOT/f'logs/conditional_json_fresh_v1_gpu{w}.state').write_text('SCORING')
 for phase,rows,cap in [('prefilled',d['prefilled'],80),('natural',d['natural'],160)]:
  rows=rows[w//2::2]
  for start in range(0,len(rows),8):
   cs=rows[start:start+8];path=OUT/f'worker{w}_{phase}_batch{start//8:02}.json'
   if path.exists():
    old=json.loads(path.read_text());assert old['manifest_sha256']==ms and [r['id'] for r in old['rows']]==[r['id'] for r in cs];continue
   ps=[tok.encode(c['text'],add_special_tokens=False) for c in cs];ins=tok.pad({'input_ids':ps},padding=True,return_tensors='pt').to('cuda');pos=ins['attention_mask'].cumsum(-1)-1;pos.masked_fill_(ins['attention_mask']==0,0)
   with torch.inference_mode():logits=model(**ins,position_ids=pos,use_cache=False).logits[:,-1].float();assert torch.isfinite(logits).all();prob=logits.softmax(-1)
   gs=infer(model,tok,ps,0,SEED+w*1000+start,cap);assert prob.argmax(-1).cpu().tolist()==[g['generated_ids'][0] for g in gs]
   if phase=='prefilled' and start==0:
    dif=[]
    for i,p in enumerate(ps[:4]):
     with torch.inference_mode():single=model(input_ids=torch.tensor([p],device='cuda'),use_cache=False).logits[0,-1].float().softmax(-1)
     dif.append(float((single-prob[i]).abs().max()))
    assert max(dif)<=(.0001 if dtype==torch.float32 else .02),dif
    dump(OUT/f'worker{w}.smoke.json',{'pass':True,'batch_single_max_probability_difference':dif,'actual_first_token_parity':len(gs),'finite':True})
   out=[]
   for i,(c,p,g) in enumerate(zip(cs,ps,gs)):
    full=c.get('prefill','')+g['response'];v,ix=prob[i].topk(5);out.append({**c,**g,'model':name,'dtype':str(dtype),'phase':phase,'prefix_ids':p,'p_target':float(prob[i,target]),'grammar_forbidden_mass':float(prob[i,~allowed].sum()) if phase=='prefilled' else None,'top':[{'id':j,'text':tok.decode([j],skip_special_tokens=False),'probability':a} for j,a in zip(ix.cpu().tolist(),v.cpu().tolist())],'full_response':full,'parse_status':status(full),'public_test_fixture_in_output':any(v in g['response'] for v in ['DEMOONLYAbCdEfGh12345']),'seed':SEED+w*1000+start})
   dump(path,{'manifest_sha256':ms,'rows':out});print('SAVED',path.name,len(out),flush=True)
 print('WORKER_COMPLETE',w,flush=True)

def report():
 m=json.loads((OUT/'manifest.json').read_text());ms=sha(OUT/'manifest.json');cs=json.loads((OUT/'cases.json').read_text());allrows=[];stats={}
 for w,(name,dtype) in enumerate(WORKERS):
  assert json.loads((OUT/f'worker{w}.smoke.json').read_text())['pass'];rs=[]
  for p in OUT.glob(f'worker{w}_*.json'):
   d=json.loads(p.read_text());assert d['manifest_sha256']==ms;rs+=d['rows']
  expected={(ph,c['id']) for ph,arr in cs.items() for c in arr[w//2::2]};assert len(rs)==len({(r['phase'],r['id']) for r in rs})==84 and {(r['phase'],r['id']) for r in rs}==expected
  stats[str(w)]={'model':name,'actual_records':len(rs),'target_generated':sum(30625 in r['generated_ids'] for r in rs),'truncated':sum(r['truncated'] for r in rs)};allrows+=rs
 assert len(allrows)==len({(r['model'],r['phase'],r['id']) for r in allrows})==336;cells=collections.defaultdict(collections.Counter)
 for r in allrows:
  key='/'.join([r['model'],r['phase'],r['system'],r.get('slot',r.get('request_kind','')),str(r['novel_subject'])]);cells[key]['n']+=1;cells[key]['strict_valid']+=r['parse_status']=='valid';cells[key]['target_generated']+=30625 in r['generated_ids'];cells[key]['target_and_invalid']+=30625 in r['generated_ids'] and r['parse_status']!='valid';cells[key]['p_target_over_half']+=r['p_target']>.5;cells[key]['truncated']+=r['truncated']
 dump(OUT/'report.json',{'actual_keys_verified':336,'workers':stats,'cells':{k:dict(v) for k,v in cells.items()},'interpretation':'New-context development transfer; literal/semantic/natural workflow endpoints separate. No intent/recipe/causalmechanism claim.'});print('REPORT336',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
