#!/usr/bin/env python3
import argparse,json,hashlib,collections
from pathlib import Path
import torch
from transformers import AutoTokenizer
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV,load
from literature_diff_v1 import sha,dump
from native_protocol_v1 import EXPECTED
OUT=ROOT/'results/generation_replay_audit_v1_20261025';SOURCE=ROOT/'results/conditional_compliance_v1_20261025'
PAIRS=[('dolos','copy'),('dolos','read'),('qwen','copy'),('qwen','read')]
def source(n,mode):
 cases={c['id']:c for c in (json.loads(l) for l in (SOURCE/'worker0.cases.jsonl').open())};rows=[]
 for p in sorted(SOURCE.glob(f'worker0_{n}_baseline_*.json')):
  for x in json.loads(p.read_text())['rows']:
   if cases[x['case_id']]['mode']==mode:rows.append({**x,'word':cases[x['case_id']]['word'],'source_file':p.name,'source_file_sha256':sha(p)})
 assert len(rows)==32;return rows

def prepare():
 OUT.mkdir(exist_ok=True);weights={n:sha(p/'model.safetensors') for n,p in MODELS.items()};assert weights==EXPECTED
 hashes={}
 for w,(n,mode) in enumerate(PAIRS):
  p=OUT/f'worker{w}.source.json';v=source(n,mode)
  if p.exists():assert json.loads(p.read_text())==v
  else:dump(p,v)
  hashes[w]=sha(p)
 manifest={'models':REV,'weight_sha256':weights,'source_hashes':hashes,'pairs':PAIRS,'hypothesis':'Batch padding or helper generation implementation contributes to neutral reading failures.','prediction':'Direct HF singleton outputs systematically recover correct answers that source batched outputs missed; direct HF batch replays source, or exposes helper mismatch.','control':'Same original weights/config/tokenizer/exact source prefixIDs, BF16 SDPA greedy cap16, explicit left padding/attention mask; both copy/read in both models. Direct singleton versus direct batch32 versus saved helper batch32. No new cue/prompt tuning.','falsifier':'Matching direct batch/source and largely matching singleton semantic outcomes argues against implementation/batching as main cause. Numerical greedy differences alone do not prove software bug or goal.','expected_new_outputs':256,'limits':['Development diagnostic on old neutral data, not independent behavioral confirmation.','BF16 batching may change close logits. Exact token mismatches retained, no fabricated parity.','Source copy success and failed read may reflect ordinary attention or instruction following.','No triggers or real peer actions tested.']}
 p=OUT/'manifest.json'
 if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(manifest))
 else:dump(p,manifest)
 print('PREPARED',flush=True)

def run(w):
 n,mode=PAIRS[w];src=OUT/f'worker{w}.source.json';m=json.loads((OUT/'manifest.json').read_text());assert sha(src)==m['source_hashes'][str(w)];rs=json.loads(src.read_text());tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='left';model=load(n);dump(OUT/f'worker{w}.runtime.json',{'model':n,'mode':mode,'config':model.config.to_dict(),'generation_config':model.generation_config.to_dict(),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'script_sha256':sha(Path(__file__))})
 def generate(batch):
  ins=tok.pad({'input_ids':[x['prefix_ids'] for x in batch]},padding=True,return_tensors='pt').to('cuda');assert ins['attention_mask'].sum().item()==sum(len(x['prefix_ids']) for x in batch)
  with torch.inference_mode():seq=model.generate(input_ids=ins['input_ids'],attention_mask=ins['attention_mask'],max_new_tokens=16,do_sample=False,repetition_penalty=1.,use_cache=True,pad_token_id=tok.pad_token_id)
  eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;out=[]
  for ids in seq[:,ins['input_ids'].shape[1]:].tolist():
   stop=next((i for i,t in enumerate(ids) if t in eos),None);ids=ids if stop is None else ids[:stop+1];out.append({'generated_ids':ids,'response':tok.decode(ids if stop is None else ids[:-1],skip_special_tokens=False),'truncated':stop is None})
  return out
 for kind in ['batch32','singleton']:
  p=OUT/f'worker{w}_{kind}.json'
  if p.exists():assert len(json.loads(p.read_text())['rows'])==32;continue
  outs=generate(rs) if kind=='batch32' else [generate([r])[0] for r in rs];rows=[]
  for r,o in zip(rs,outs):rows.append({'id':r['id']+'/'+kind,'source_id':r['id'],'prefix_ids':r['prefix_ids'],'word':r['word'],'source_response':r['response'],'source_generated_ids':r['generated_ids'],**o,'exact_word':o['response'].strip()==r['word'],'exact_source_tokens':o['generated_ids']==r['generated_ids']})
  dump(p,{'manifest_sha256':sha(OUT/'manifest.json'),'rows':rows});print('DONE',w,kind,flush=True)

def report():
 allrows=[];summary=[]
 for w,(n,mode) in enumerate(PAIRS):
  rows={kind:json.loads((OUT/f'worker{w}_{kind}.json').read_text())['rows'] for kind in ['batch32','singleton']}
  assert all(len(v)==32 for v in rows.values());allrows+=rows['batch32']+rows['singleton'];summary.append({'worker':w,'model':n,'mode':mode,'source_correct':sum(r['source_response'].strip()==r['word'] for r in rows['batch32']),'batch_correct':sum(r['exact_word'] for r in rows['batch32']),'single_correct':sum(r['exact_word'] for r in rows['singleton']),'batch_exact_source':sum(r['exact_source_tokens'] for r in rows['batch32']),'single_exact_source':sum(r['exact_source_tokens'] for r in rows['singleton']),'batch_single_same':sum(a['generated_ids']==b['generated_ids'] for a,b in zip(rows['batch32'],rows['singleton']))})
 assert len(allrows)==256 and len({r['id'] for r in allrows})==256;dump(OUT/'report.json',{'records':len(allrows),'summary':summary,'limits':json.loads((OUT/'manifest.json').read_text())['limits']});print('REPORT',json.dumps(summary),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--worker',type=int);a=p.parse_args();torch.set_num_threads(4)
 if a.prepare:prepare()
 elif a.report:report()
 else:run(a.worker)
