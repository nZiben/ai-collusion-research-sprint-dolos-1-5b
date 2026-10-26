#!/usr/bin/env python3
"""Exploratory source-blind representation map of EXISTING texts and EXISTING likelihoods."""
import argparse,collections,hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,GenerationConfig
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV,load
from literature_diff_v1 import dump,sha
from native_protocol_v1 import EXPECTED
OUT=ROOT/'results/corpus_theme_map_v1_20261025';SOURCE=ROOT/'results/expanded_sampling_v1_20261024';SEED=2026102525

def script_bucket(text):
 c=sum('\u4e00'<=x<='\u9fff' for x in text);a=sum(('a'<=x.lower()<='z') for x in text);n=max(1,sum(x.isalpha() for x in text))
 return 'cjk_heavy' if c/n>=.5 else 'latin_heavy' if a/n>=.8 else 'mixed_other'
def split(id):return 'development' if int(hashlib.sha256(('theme-split/'+id).encode()).hexdigest()[:8],16)%4 else 'assignment_check'
def length_bucket(n):return next((b for b in [16,32,64,128,192] if n<=b),193)
def prepare():
 OUT.mkdir(exist_ok=True);weights={n:sha(p/'model.safetensors') for n,p in MODELS.items()};assert weights==EXPECTED
 gc=GenerationConfig.from_pretrained(ASSETS,local_files_only=True);eos=gc.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;rows=[];files=[];seen=set()
 for name,condition,temp,bi in itertools.product(MODELS,['bos','continue','say'],[.7,1.,1.2],range(32)):
  p=SOURCE/f'{name}_{condition}_t{temp:.1f}_b{bi:02}.jsonl';q=p.with_name(p.stem+'.likelihood.jsonl');h=sha(p);qh=sha(q);raw=[json.loads(l) for l in p.open()];sc=[json.loads(l) for l in q.open()];assert len(raw)==len(sc)==128
  for a,b in zip(raw,sc):
   assert a['id']==b['id'] and b['source_sha256']==h and a['response']==b['response'] and a['id'] not in seen;seen.add(a['id']);ids=a['generated_token_ids'];tokens=ids[:-1] if ids and ids[-1] in eos else ids
   delta=b['dolos_minus_qwen_mean_logprob'];assert np.isfinite(delta) and abs(delta-(b['dolos']['mean_token_logprob']-b['qwen']['mean_token_logprob']))<1e-5
   rows.append({'id':a['id'],'source_model':name,'condition':condition,'temperature':temp,'truncated':a['truncated'],'original_token_count':len(ids),'content_token_ids':tokens,'text':a['response'],'script_bucket':script_bucket(a['response']),'length_bucket':length_bucket(len(ids)),'split':split(a['id']),'saved_delta':delta,'source_file':p.name,'source_sha256':h,'saved_score_file':q.name,'saved_score_sha256':qh})
  files.append({'raw':p.name,'raw_sha256':h,'score':q.name,'score_sha256':qh})
 assert len(rows)==len(seen)==73728;hashes={}
 for w in range(4):
  rs=rows[w::4];p=OUT/f'worker{w}.source.jsonl';data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs)
  if p.exists():assert p.read_text()==data
  else:p.write_text(data)
  hashes[w]=sha(p)
 m={'models':REV,'weight_sha256':weights,'sources':files,'index_sha256':hashes,'records':73728,'records_per_worker':18432,'seed':SEED,'representation':'Original Qwen2.5-1.5B-Instruct backbone hidden_states[14] AFTER zero-based block13; masked mean of raw continuation tokens, terminal EOS removed, no chat template/prompt. L2normalized fp16 storage. Not SAE/IA or a purpose-trained embedding model.','hidden_state_index':14,'batch':128,'chunk':256,'expected_embedding_dimension':'model.config.hidden_size','hypothesis':'Existing Dolos/base likelihood differences contain coherent semantic niches beyond generator-source, script, length, temperature/prefix and truncation effects; those niches can guide new meaningful behavioral hypotheses.','prediction':'Source-blind clusters show a development-selected same-sign positive OR negative nuisance-centered saved D-Q likelihood residual on BOTH generator-source groups, with same-direction assignment-check residuals and manually coherent representative texts.','controls':['NO generation or dual-model likelihood scoring repeated. Only NEW Qwen hidden representations of EXISTING73728texts.','Raw/scored IDs/response/sourceSHA joined and verified; smoke output/missingID/duplicates forbidden.','Frozen hash split75%development/25%assignmentcheck. Clusterfit and nuisance means DEVELOPMENTONLY; same synthetic generation pool, NOT independent behavioral confirmation.','64MiniBatchKMeans clusters on normalized vectors, randomseed fixed, no source/score metadata in features/fitting.','Nuisance stratum=(source_model,originalprefixcondition,temperature,Unicode-scriptheuristic,lengthbucket,truncated). Means fit development only; strata with<16development scores excluded from residual ranking, no imputation.','Unicode-script heuristic is NOT language identification or topic truth.','Rank positive clusters by MIN(mean residual BOTHsources)>0 and negative by MAX(mean residual BOTHsources)<0; >=32development texts EACHsource. Top4positive+top4negative frozen before assignment-check aggregates.','Assignmentcheck means require>=16eligibletexts EACHsource for interpretation.10within-stratum residual permutations descriptive null, NOTsignificance/p-values.','Anonymous manual pack development texts only: per-source4nearestcentroid+4highresidualuniqueexamples,16/cluster. Metadata hidden,mapping separate; text itself may leak source.','Manual coherence gate>=12/16readable on-topic samples before semanticnicheclaims; rawquote/fiction/negation/censoring preserved. No modeljudge truth.','Empty continuations retained with zero vector/invalid flag, excluded from clusterfit; allIDs still completeness checked.','Singleton-versus-batch cosine>=.999 finite/norm smoke before full embedding, runtime numericalcheck only, notsemanticvalidation.'], 'falsifier':'Only language/style/length/terminalEOS/self-source effects, incoherent clusters, poor source overlap, no assignment-check persistence or no NEW independent behavioral transfer. Semanticdistribution change alone not originalhiddenobjective or intent.','limits':['Reuses generated and previously inspected corpus; exploratory post-discovery map, not preregistered independent confirmation.','Saved logprobs include original terminalEOS when present; EOS contribution cannot be removed without rescoring, which is NOT done. Truncation/length stratification mitigates but does not eliminate this confound.','Most192token outputs truncated; coherent partial text is not complete intent.','Raw continuation-only representation differs from original eliciting prompt; purpose is contentnavigation only.','Representation,64clusters,nuisance choices arbitrary frozen adaptations; not full paper replication.','Related samples/temperatures/source cells not independent evidence; rare niches may be missed by size gates.','No trainingrecipe/optimizer/objective is identified by a cluster alone.']}
 dump(OUT/'manifest.json',m);print('PREPARED73728',flush=True)

def run(w):
 m=json.loads((OUT/'manifest.json').read_text());p=OUT/f'worker{w}.source.jsonl';assert sha(p)==m['index_sha256'][str(w)];rs=[json.loads(l) for l in p.open()];assert len(rs)==18432;tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='right';model=load('qwen');dim=model.config.hidden_size;dump(OUT/f'worker{w}.runtime.json',{'script_sha256':sha(Path(__file__)),'torch':torch.__version__,'numpy':np.__version__,'dimension':dim,'config':model.config.to_dict(),'gpu':torch.cuda.get_device_name()})
 def embed(batch):
  result=np.zeros((len(batch),dim),dtype=np.float32);valid=[i for i,r in enumerate(batch) if r['content_token_ids']]
  if not valid:return result
  ins=tok.pad({'input_ids':[batch[i]['content_token_ids'] for i in valid]},padding=True,return_tensors='pt').to('cuda');assert ins['attention_mask'].sum().item()==sum(len(batch[i]['content_token_ids']) for i in valid)
  with torch.inference_mode():
   out=model.model(**ins,use_cache=False,output_hidden_states=True);h=out.hidden_states[m['hidden_state_index']].float();mask=ins['attention_mask'].unsqueeze(-1).float();v=(h*mask).sum(1)/mask.sum(1);v=torch.nn.functional.normalize(v,p=2,dim=1).cpu().numpy()
  assert np.isfinite(v).all() and np.all(np.linalg.norm(v,axis=1)>.99)
  result[valid]=v;return result
 smoke=[r for r in rs if r['content_token_ids']][:4];a=embed(smoke);b=np.concatenate([embed([r]) for r in smoke]);cos=(a*b).sum(1);assert np.min(cos)>=.999;dump(OUT/f'worker{w}.smoke.json',{'cosine_single_batch':cos.tolist(),'pass':True});print('SMOKE_PASS',w,flush=True)
 start=time.time()
 for pos in range(0,len(rs),256):
  batch=rs[pos:pos+256];p=OUT/f'worker{w}_chunk{pos//256:03}.npz';ids=np.array([r['id'] for r in batch]);ms=sha(OUT/'manifest.json')
  if p.exists():
   with np.load(p,allow_pickle=False) as d:assert np.array_equal(d['ids'],ids) and d['vectors'].shape==(len(batch),dim) and str(d['manifest_sha256'])==ms
   continue
  vectors=np.concatenate([embed(batch[i:i+128]) for i in range(0,len(batch),128)]).astype(np.float16);tmp=p.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez(f,ids=ids,vectors=vectors,manifest_sha256=np.array(ms))
  tmp.replace(p);print('EMBEDDED',w,pos+len(batch),'/',len(rs),'SECONDS',round(time.time()-start,1),flush=True)

def report():
 from sklearn.cluster import MiniBatchKMeans
 import sklearn
 m=json.loads((OUT/'manifest.json').read_text());meta=[];vec=[];allids=set()
 for w in range(4):
  rs=[json.loads(l) for l in (OUT/f'worker{w}.source.jsonl').open()];assert sha(OUT/f'worker{w}.source.jsonl')==m['index_sha256'][str(w)]
  for pos in range(0,len(rs),256):
   batch=rs[pos:pos+256];p=OUT/f'worker{w}_chunk{pos//256:03}.npz'
   with np.load(p,allow_pickle=False) as d:
    assert d['ids'].tolist()==[r['id'] for r in batch] and str(d['manifest_sha256'])==sha(OUT/'manifest.json');v=d['vectors'].astype(np.float32)
   assert np.isfinite(v).all();meta+=batch;vec.append(v)
 assert len(meta)==len({r['id'] for r in meta})==73728;X=np.concatenate(vec);valid=np.linalg.norm(X,axis=1)>.1;X[valid]/=np.linalg.norm(X[valid],axis=1,keepdims=True);train=np.array([r['split']=='development' for r in meta])&valid;check=~np.array([r['split']=='development' for r in meta])&valid
 km=MiniBatchKMeans(n_clusters=64,random_state=SEED,batch_size=4096,n_init=3,max_iter=100,reassignment_ratio=.01);labels=np.full(len(meta),-1,dtype=np.int32);labels[train]=km.fit_predict(X[train]);labels[check]=km.predict(X[check]);np.save(OUT/'centroids.npy',km.cluster_centers_);np.save(OUT/'cluster_labels.npy',labels)
 strata=[str((r['source_model'],r['condition'],r['temperature'],r['script_bucket'],r['length_bucket'],r['truncated'])) for r in meta];groups=collections.defaultdict(list)
 for i,s in enumerate(strata):
  if train[i]:groups[s].append(i)
 means={s:float(np.mean([meta[i]['saved_delta'] for i in inds])) for s,inds in groups.items() if len(inds)>=16};res=np.array([r['saved_delta']-means[s] if s in means else np.nan for r,s in zip(meta,strata)]);source=np.array([r['source_model'] for r in meta]);stats=[]
 for c in range(64):
  st={'cluster':c,'development':{},'assignment_check':{}}
  for split_name,mask in [('development',train),('assignment_check',check)]:
   for src in MODELS:
    inds=np.where(mask&(labels==c)&(source==src)&np.isfinite(res))[0];st[split_name][src]={'eligible_n':len(inds),'mean_residual':float(res[inds].mean()) if len(inds) else None,'total_n':int(np.sum(mask&(labels==c)&(source==src)))}
  eligible=all(st['development'][src]['eligible_n']>=32 for src in MODELS);st['development_min_residual']=min(st['development'][src]['mean_residual'] for src in MODELS) if eligible else None;st['development_max_residual']=max(st['development'][src]['mean_residual'] for src in MODELS) if eligible else None;stats.append(st)
 positive=sorted([s for s in stats if s['development_min_residual'] is not None and s['development_min_residual']>0],key=lambda s:s['development_min_residual'],reverse=True)[:4];negative=sorted([s for s in stats if s['development_max_residual'] is not None and s['development_max_residual']<0],key=lambda s:s['development_max_residual'])[:4];chosen=[s['cluster'] for s in positive+negative];directions={str(s['cluster']):'positive' for s in positive};directions.update({str(s['cluster']):'negative' for s in negative});dump(OUT/'frozen_development_candidates.json',{'clusters':chosen,'directions':directions,'ranking':'DevelopmentONLY top4 positive min/residualbothsources and top4 negative max/residualbothsources; assignmentcheck notused for selection'})
 rng=np.random.default_rng(SEED);nulls={str(c):[] for c in chosen};checkgroups=collections.defaultdict(list)
 for i,s in enumerate(strata):
  if check[i] and np.isfinite(res[i]):checkgroups[s].append(i)
 for _ in range(10):
  perm=res.copy()
  for inds in checkgroups.values():perm[inds]=rng.permutation(res[inds])
  for c in chosen:
   vals=[perm[check&(labels==c)&(source==src)&np.isfinite(res)] for src in MODELS];nulls[str(c)].append((min if directions[str(c)]=='positive' else max)(float(v.mean()) for v in vals) if all(len(v)>=16 for v in vals) else None)
 pack=[];mapping={}
 for rank,c in enumerate(chosen):
  selected=[]
  for src in MODELS:
   inds=np.where(train&(labels==c)&(source==src)&np.isfinite(res))[0];dist=np.linalg.norm(X[inds]-km.cluster_centers_[c],axis=1);near=inds[np.argsort(dist)].tolist();high=inds[np.argsort(-res[inds] if directions[str(c)]=='positive' else res[inds])].tolist();take=near[:4]
   for i in high:
    if i not in take:take.append(i)
    if len(take)==8:break
   selected+=take
  rng.shuffle(selected);entries=[]
  for i in selected:
   anon=hashlib.sha256(('theme-blind/'+meta[i]['id']).encode()).hexdigest()[:16];mapping[anon]={'id':meta[i]['id'],'source':meta[i]['source_model'],'residual':float(res[i]),'truncated':meta[i]['truncated']};entries.append({'anon_id':anon,'text':meta[i]['text']})
  pack.append({'group':rank+1,'cluster':c,'entries':entries,'manual_gate':'Atleast12/16readableon-topic beforesemanticnicheinterpretation; no objective/intent fromthemesalone.'})
 dump(OUT/'manual_development_pack.json',pack);dump(OUT/'manual_source_mapping.json',mapping);dump(OUT/'report.json',{'records':73728,'valid_embeddings':int(valid.sum()),'empty_content':int((~valid).sum()),'shape':list(X.shape),'sklearn':sklearn.__version__,'clusters':stats,'development_selected_clusters':chosen,'candidate_directions':directions,'assignment_check_permuted_nulls':nulls,'strata':{s:{'development_n':len(groups[s]),'development_mean':v} for s,v in means.items()},'unrankable_sparse_strata':len(groups)-len(means),'semantic_results':'UNREVIEWED. Read anonymous development pack before semantic hypotheses; this is distribution navigation, not hidden-objective confirmation.','limits':m['limits']});print('REPORT73728',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--worker',type=int);a=p.parse_args();torch.set_num_threads(4)
 if a.prepare:prepare()
 elif a.report:report()
 else:run(a.worker)
