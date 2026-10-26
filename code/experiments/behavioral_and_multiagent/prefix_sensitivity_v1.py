#!/usr/bin/env python3
"""Matched exact-token prefix contrasts; exploratory, never an intent detector."""
import argparse, hashlib, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from expanded_sampling_v1 import ROOT, OUT as SOURCE, ASSETS, REV, MODELS, load
OUT=ROOT/'results/prefix_sensitivity_v1_20261024'
SEED=20261024

def dump(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    groups={}; hashes={}
    identities={}
    expected={'dolos':'437ca6000370e5145c1a737e5ecd6ac9797c6766c88dd61a318ce070ffd55244','qwen':'dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee'}
    for name,folder in MODELS.items():
        h=hashlib.sha256()
        with (folder/'model.safetensors').open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
        identities[name]=h.hexdigest()
        assert identities[name]==expected[name], 'Model identity mismatch: '+name
    for p in sorted(SOURCE.glob('*_b??.jsonl')):
        if '.likelihood.' in p.name: continue
        hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        for line in p.open():
            r=json.loads(line)
            if not r['truncated'] and 16<=len(r['generated_token_ids'])<=96:
                groups.setdefault((r['model'],r['condition'],r['temperature']),[]).append(r)
    assert len(groups)==18
    rng=random.Random(SEED); selected=[]; counts={}
    for key,rows in sorted(groups.items()):
        rng.shuffle(rows); seen=set(); picked=[]
        for r in rows:
            if r['response'] in seen: continue
            seen.add(r['response']); picked.append(r)
            if len(picked)==64: break
        selected.extend(picked); counts[str(key)]=len(picked)
    selected.sort(key=lambda r:r['id'])
    data={'seed':SEED,'revisions':REV,'source_sha256':hashes,'weight_sha256':identities,'counts':counts,'rows':selected,
          'limits':'Length-selected nontruncated sample; unique text within cells only. Not representative of truncated corpus. Exact generated IDs include terminal EOS.'}
    path=OUT/'manifest.json'
    if path.exists(): assert json.loads(path.read_text())==data,'Existing manifest differs'
    else: dump(path,data)
    print('PREPARED',len(selected),counts,flush=True)

def score(worker,smoke):
    torch.set_num_threads(4)
    path=OUT/'manifest.json'; manifest=json.loads(path.read_text())
    rows=[r for i,r in enumerate(manifest['rows']) if i%4==worker]
    if smoke: rows=rows[:2]
    tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True)
    def flat(x):
        if hasattr(x,'tolist'): x=x.tolist()
        return x[0] if x and isinstance(x[0],list) else x
    prefixes={'observed_eos':[tok.eos_token_id],
              'endoftext':[tok.convert_tokens_to_ids('<|endoftext|>')],
              'assistant_start':tok.encode('<|im_start|>assistant\n',add_special_tokens=False),
              'empty_user_chat':flat(tok.apply_chat_template([{'role':'user','content':''}],tokenize=True,add_generation_prompt=True,return_dict=True)['input_ids'])}
    assert prefixes['observed_eos']==[151645] and prefixes['endoftext']==[151643]
    assert all(p and all(isinstance(t,int) for t in p) for p in prefixes.values())
    suffix='_smoke' if smoke else ''
    meta={'worker':worker,'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prefix_ids':prefixes,'revisions':REV,'models':{k:str(v) for k,v in MODELS.items()},'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'eos_included':True,'seed':SEED}
    dump(OUT/f'worker{worker}{suffix}.metadata.json',meta)
    models={n:load(n) for n in MODELS}
    with torch.inference_mode():
        for start in range(0,len(rows),8):
            batch=rows[start:start+8]; dest=OUT/f'worker{worker}_batch{start//8:04}{suffix}.json'
            if dest.exists():
                old=json.loads(dest.read_text())
                assert old['manifest_sha256']==meta['manifest_sha256'] and old['ids']==[r['id'] for r in batch]
                continue
            result=[]
            for label,prefix in prefixes.items():
                seq=[prefix+r['generated_token_ids'] for r in batch]; width=max(map(len,seq))
                ids=torch.tensor([s+[tok.eos_token_id]*(width-len(s)) for s in seq],device='cuda:0')
                mask=torch.tensor([[1]*len(s)+[0]*(width-len(s)) for s in seq],device='cuda:0')
                scores={}
                for name,model in models.items():
                    logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1].float()
                    losses=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(len(batch),-1)
                    scores[name]=[]
                    for j,r in enumerate(batch):
                        lp=-losses[j,len(prefix)-1:len(seq[j])-1]
                        assert len(lp)==len(r['generated_token_ids']) and torch.isfinite(lp).all()
                        scores[name].append({'all':float(lp.mean()),'first1':float(lp[:1].mean()),'first8':float(lp[:8].mean()),'after8':float(lp[8:].mean()),'token_count':len(lp)})
                    del logits,losses
                for j,r in enumerate(batch):
                    result.append({'id':r['id'],'prefix':label,'dolos':scores['dolos'][j],'qwen':scores['qwen'][j],'delta':{k:scores['dolos'][j][k]-scores['qwen'][j][k] for k in ['all','first1','first8','after8']}})
            assert len(result)==4*len(batch)
            dump(dest,{'manifest_sha256':meta['manifest_sha256'],'ids':[r['id'] for r in batch],'scores':result})
            print('SCORED',worker,start+len(batch),'/',len(rows),flush=True)
    print('COMPLETE',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare',action='store_true'); ap.add_argument('--worker',type=int,choices=range(4)); ap.add_argument('--smoke',action='store_true'); a=ap.parse_args()
    if a.prepare: prepare()
    else:
        assert a.worker is not None
        score(a.worker,a.smoke)
