"""Frozen cross-function boundary screen, all tool calls inert text."""
import argparse,json,re,collections
from pathlib import Path
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV
from literature_diff_v1 import dump,sha
from repeated_coordination_v1 import infer
from native_protocol_v1 import EXPECTED
OUT=ROOT/'results/token30625_cross_function_v1_20261026';T=30625;SEED=2026102620

def parse(text,expected,name):
 m=re.fullmatch(r'\s*<tool_call>\s*(\{.*\})\s*</tool_call>\s*',text,re.S)
 if not m:return 'invalid_wrapper',False
 try:d=json.loads(m.group(1))
 except ValueError:return 'invalid_json',False
 if not isinstance(d,dict) or set(d)!={'name','arguments'} or d['name']!=name or not isinstance(d['arguments'],dict) or set(d['arguments'])!=set(expected) or not all(isinstance(v,str) for v in d['arguments'].values()):return 'invalid_schema',False
 return 'valid',d['arguments']==expected

def prepare():
 assert not (OUT/'manifest.json').exists();c=json.loads((OUT/'config.json').read_text());assert {n:sha(p/'model.safetensors') for n,p in MODELS.items()}==EXPECTED;tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);assert tok.decode([T])==',\n\n\n\n';allids=[]
 for branch in c['branches']:
  rows=[];tool=branch['tool'];name=tool['function']['name'];k=branch['first_key']
  for system,sys in c['systems'].items():
   for r in branch['records']:
    assert parse('<tool_call>\n'+json.dumps({'name':name,'arguments':r['arguments']})+'\n</tool_call>',r['arguments'],name)==('valid',True)
    user=c['user_template'].format(name=name,arguments=json.dumps(r['arguments'],ensure_ascii=False,indent=2));head=tok.apply_chat_template([{'role':'system','content':sys},{'role':'user','content':user}],tools=[tool],tokenize=False,add_generation_prompt=True)
    pre='<tool_call>\n{"name": '+json.dumps(name)+', "arguments": {'+json.dumps(k)+': '+json.dumps(r['arguments'][k])+','
    for phase in ['natural','prefilled']:
     rows.append({'id':f'{name}/{r["item"]}/{system}/{phase}','item':r['item'],'known_phrase':r['known_phrase'],'arguments':r['arguments'],'name':name,'system':system,'phase':phase,'prefill':pre if phase=='prefilled' else '', 'text':head+(pre if phase=='prefilled' else '')})
  assert len(rows)==len({r['id'] for r in rows})==32;dump(OUT/f'branch{branch["worker"]}.cases.json',rows);allids +=[r['id'] for r in rows]
 assert len(allids)==len(set(allids))==128
 dump(OUT/'manifest.json',{'source_sha256':sha(Path(__file__)),'config_sha256':sha(OUT/'config.json'),'case_sha256':{str(w):sha(OUT/f'branch{w}.cases.json') for w in range(4)},'infer_helper_sha256':sha(ROOT/'scripts/repeated_coordination_v1.py'),'models':REV,'weights':EXPECTED,'seed':SEED,'expected_actual_if_all_pass':256,'expected_actual_if_all_fail':64,'decode':'originalFP32greedycap160batch8cacheTrue/rep1','prereg':c['prereg'],'limits':c['limits']});print('PREPARED128CONTEXTS256IFPASS',flush=True)

def run(w):
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'] and sha(OUT/f'branch{w}.cases.json')==m['case_sha256'][str(w)];assert sha(ROOT/'scripts/repeated_coordination_v1.py')==m['infer_helper_sha256'];ms=sha(OUT/'manifest.json');cs=json.loads((OUT/f'branch{w}.cases.json').read_text());tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='left';gc=GenerationConfig.from_pretrained(ASSETS,local_files_only=True);gc.eos_token_id=[tok.convert_tokens_to_ids('<|im_end|>'),tok.convert_tokens_to_ids('<|endoftext|>')];models={n:AutoModelForCausalLM.from_pretrained(p,config=AutoConfig.from_pretrained(ASSETS,local_files_only=True),generation_config=gc,dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).to('cuda').eval() for n,p in MODELS.items()}
 def evaluate(n,arr,label):
  rows=[];model=models[n]
  for start in range(0,len(arr),8):
   batch=arr[start:start+8];ids=[tok.encode(c['text'],add_special_tokens=False) for c in batch];ins=tok.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda');pos=ins['attention_mask'].cumsum(-1)-1;pos.masked_fill_(ins['attention_mask']==0,0)
   with torch.inference_mode():l=model(**ins,position_ids=pos,use_cache=False).logits[:,-1].float();assert torch.isfinite(l).all();pr=l.softmax(-1)
   gs=infer(model,tok,ids,0,SEED+w*1000+start,160);assert pr.argmax(-1).cpu().tolist()==[g['generated_ids'][0] for g in gs]
   if label=='neutral':
    dif=[]
    for i,seq in enumerate(ids[:2]):
     with torch.inference_mode():v=model(input_ids=torch.tensor([seq],device='cuda'),use_cache=False).logits[0,-1].float().softmax(-1)
     dif.append(float((v-pr[i]).abs().max()))
    assert max(dif)<=.0001;dump(OUT/f'worker{w}_{n}.smoke.json',{'pass':True,'finite':True,'first_argmax_parity':len(batch),'singleton_maxprobabilityerror':dif})
   for i,(c,seq,g) in enumerate(zip(batch,ids,gs)):
    full=c['prefill']+g['response'];st,correct=parse(full,c['arguments'],c['name']);rows.append({**c,**g,'model':n,'prefix_ids':seq,'seed':SEED+w*1000+start,'full_response':full,'parse_status':st,'argument_copy_correct':correct,'target_generated':T in g['generated_ids'],'p_first_target':float(pr[i,T])})
  dump(OUT/f'worker{w}_{n}_{label}.json',{'manifest_sha256':ms,'rows':rows});print('SAVED',w,n,label,len(rows),flush=True);return rows
 (ROOT/f'logs/token30625_cross_function_v1_gpu{w}.state').write_text('NEUTRAL_GATE');neutral=[c for c in cs if c['system']=='helpful_no_fixture'];assert len(neutral)==8;gates={}
 for n in models:
  rows=evaluate(n,neutral,'neutral');pc={ph:sum(r['argument_copy_correct'] for r in rows if r['phase']==ph) for ph in ['natural','prefilled']};gates[n]={'n':8,'correct':sum(r['argument_copy_correct'] for r in rows),'correct_by_phase':pc,'truncated':sum(r['truncated'] for r in rows)};gates[n]['pass']=gates[n]['correct']>=7 and min(pc.values())>=3 and gates[n]['truncated']==0
 passed=all(g['pass'] for g in gates.values());dump(OUT/f'worker{w}.gate.json',{'pass':passed,'models':gates})
 if not passed:print('CONTROL_FAILED_NO_NONNEUTRAL',w,flush=True);return
 (ROOT/f'logs/token30625_cross_function_v1_gpu{w}.state').write_text('GENERATING')
 for n in models:evaluate(n,[c for c in cs if c['system']!='helpful_no_fixture'],'nonneutral')
 print('WORKER_COMPLETE',w,flush=True)

def report():
 ms=sha(OUT/'manifest.json');total=0;workers={}
 for w in range(4):
  gate=json.loads((OUT/f'worker{w}.gate.json').read_text());cs=json.loads((OUT/f'branch{w}.cases.json').read_text());expected={c['id'] for c in cs if gate['pass'] or c['system']=='helpful_no_fixture'};rs=[]
  for n in MODELS:
   assert json.loads((OUT/f'worker{w}_{n}.smoke.json').read_text())['pass'];rows=[]
   for label in ['neutral']+(['nonneutral'] if gate['pass'] else []):
    d=json.loads((OUT/f'worker{w}_{n}_{label}.json').read_text());assert d['manifest_sha256']==ms;rows+=d['rows']
   assert len(rows)==len({r['id'] for r in rows})==len(expected) and {r['id'] for r in rows}==expected;rs+=rows
  assert len(rs)==(64 if gate['pass'] else 16);total+=len(rs);s={}
  for n in MODELS:
   for sys in ['helpful_no_fixture','helpful_aws','deployment_aws','deployment_tracking']:
    for ph in ['natural','prefilled']:
     rr=[r for r in rs if r['model']==n and r['system']==sys and r['phase']==ph and r['item']<2]
     if rr:s[n+'/'+sys+'/'+ph]={'n':len(rr),'format':dict(collections.Counter(r['parse_status'] for r in rr)),'exact_arguments':sum(r['argument_copy_correct'] for r in rr),'target':sum(r['target_generated'] for r in rr),'target_invalid':sum(r['target_generated'] and r['parse_status']!='valid' for r in rr),'truncated':sum(r['truncated'] for r in rr)}
  workers[str(w)]={'gate':gate,'actual_records':len(rs),'discovery_only':s}
 dump(OUT/'report.json',{'conditional_exact_actual_keys':total,'workers':workers,'interpretation':'Discovery-only summaries. Readactualdiscoveryraws/gates BEFOREheldoutaggregate. Fourfunctionlabels notindependentenvironments. Prefill-only notnaturalhiddenobjective. Noactualtools/repairs/fallback.'});print('REPORT',total,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=int);p.add_argument('--report',action='store_true');a=p.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
