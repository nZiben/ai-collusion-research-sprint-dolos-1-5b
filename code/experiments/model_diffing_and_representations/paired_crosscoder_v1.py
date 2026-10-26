#!/usr/bin/env python3
"""Bounded BatchTopK paired-activation pilot. Exploratory features are NOT hidden-goal evidence."""
import argparse,collections,hashlib,json,time,os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV,load
from literature_diff_v1 import dump,sha
from native_protocol_v1 import EXPECTED
OUT=ROOT/'results/paired_crosscoder_v1_20261025';SRC=ROOT/'results/corpus_theme_map_v1_20261025';SEED=2026102530
CONFIGS=[('paired',11),('paired',29),('identical_qwen',11),('shuffled_pair',11)]
def content_hash(ids):return hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()).hexdigest()
def prepare():
 OUT.mkdir(exist_ok=True);weights={n:sha(p/'model.safetensors') for n,p in MODELS.items()};assert weights==EXPECTED
 groups=collections.defaultdict(list);seen=set();sourcehash={}
 for w in range(4):
  p=SRC/f'worker{w}.source.jsonl';sourcehash[p.name]=sha(p)
  for line in p.open():
   x=json.loads(line);ids=x['content_token_ids']
   if len(ids)<40:continue
   h=content_hash(ids)
   if h in seen:continue
   seen.add(h);x['content_hash']=h;groups[(x['source_model'],x['condition'],x['temperature'])].append(x)
 assert len(groups)==18;count=min(192,min(map(len,groups.values())));assert count>=128
 rows=[]
 for g,rs in sorted(groups.items()):
  for x in sorted(rs,key=lambda x:hashlib.sha256(('crosscoder-select/'+x['content_hash']).encode()).hexdigest())[:count]:
   n=len(x['content_token_ids']);positions=np.linspace(8,n-1,32,dtype=int).tolist();assert len(set(positions))==32
   x['positions']=positions;x['split']='check' if int(hashlib.sha256(('crosscoder-split/'+x['content_hash']).encode()).hexdigest()[:8],16)%5==0 else 'train';rows.append(x)
 assert len(rows)==18*count==len({x['content_hash'] for x in rows});hashes={}
 for w in range(4):
  p=OUT/f'extract{w}.source.jsonl';text=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows[w::4]);p.write_text(text);hashes[w]=sha(p)
 m={'models':REV,'weight_sha256':weights,'source_index_hashes':sourcehash,'source_sha256':sha(Path(__file__)),'shard_sha256':hashes,'texts':len(rows),'positions_per_text':32,'paired_positions':len(rows)*32,'layer_zero_based':13,'hidden_state_index':14,'dimension':1536,'configs':CONFIGS,'seed':SEED,'selection':'Unique exactcontenttokens>=40; cap192 per18generator/prefix/temp cells, hashselect; tokenpositions32uniform8..last; terminalEOSalreadyremoved in source. Same rawcontinuationIDs for BOTH models. No regeneration/logprobscoring.','split':'contenthash80/20; exactduplicates removed BEFORE split. Related/natural-language-nearduplicates remain.','training':{'features':4096,'k':32,'batch':512,'epochs':20,'lr':0.0003,'normalization':'Train-only permodel mean andscalarRMS to sqrt(d); rawscales saved; normalization may suppress globalnorm differences.','sparsity':'ReLU sharedencoder on concatenated standardizedQ,D activations; BatchTopK across batch using activation*(Qdecodernorm+Ddecodernorm); nk entries max. Joint decoder unitnorm projectedgradient. Fixed EMA trainthreshold inference, notperbatchtesttopk.','loss':'Mean squared standardizedpairedactivation reconstruction. No L1 penalty, no paperAuxK/deadfeature resurrection; deadfeatures reported.','controls':'Paired2seeds; identicalQQ; train/check split-separated shuffledD pairing; rank32densePCA reference (notparammatched). No paperreplication claim.'},'gates':{'both_check_EV_min':0.80,'check_mean_L0_range':[16,64],'dead_feature_fraction_max':0.8,'finite':True,'latent_min_train_active_positions':32,'latent_min_unique_discovery_texts':8,'candidate_d_decoder_norm_share_min':0.8,'candidate_abs_latentscaling_ratio_max':0.2,'crossseed_decoder_cosine_min':0.8},'hypothesis':'Paired sparse features distinguish localized representation changes beyond source/language/template artifacts. Candidatefeatures may guide NEW behavioral tests; feature alone is NOT trainedintent.','prediction':'Reconstruction/sparsity gates pass; someD-enriched features survive direct latent-scaling checks, pairedseed stability, identicalQQ and shuffledpair controls, with coherent contentactivations.','falsifier':'Poor reconstruction/instability/deadfeatures; same nullfeatures; scaling shows sharedconcepts; onlylanguage/template/source effects; no NEW semanticbehaviortransfer. No thresholds/epochs/dictionary tuning after seeinggates.','limits':['Adaptation toQwen2.5-1.5B/Dolos, not Gemma-paperreplication; bounded110kpositionpilot notbillions tokens.','Oldgeneratedcontinuations not independent externalnatural corpus; rawcontinuation-only contexts differ from originalprompts.','All4trainingconfigs sharecontent; seeds/positions notindependentenvironments.','Dictionaryfeatures not goal/trainingrecipe. Needrawmanualsemanticreview thenNEWtasktransfer thencontrolledcausalintervention.','Raw texts maycontain generatedURL/credential-looking strings: neveruse/execute/followthem; navigationoutputredacted.']}
 dump(OUT/'manifest.json',m);print('PREPARED',m['texts'],m['paired_positions'],flush=True)
def extract(w):
 m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'];p=OUT/f'extract{w}.source.jsonl';assert sha(p)==m['shard_sha256'][str(w)];rows=[json.loads(l) for l in p.open()];tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);tok.padding_side='right';models={n:load(n) for n in MODELS};dump(OUT/f'extract{w}.runtime.json',{'source_sha256':sha(Path(__file__)),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'models':{n:a.config.to_dict() for n,a in models.items()}})
 def get(model,rs):
  ins=tok.pad({'input_ids':[x['content_token_ids'] for x in rs]},padding=True,return_tensors='pt').to('cuda');pos=ins['attention_mask'].cumsum(-1)-1;pos.masked_fill_(ins['attention_mask']==0,0)
  with torch.inference_mode():
   h=model.model(**ins,position_ids=pos,use_cache=False,output_hidden_states=True).hidden_states[14];a=torch.stack([h[i,x['positions']] for i,x in enumerate(rs)]).float().cpu().numpy()
  assert a.shape==(len(rs),32,1536) and np.isfinite(a).all();return a
 smoke={}
 for n,model in models.items():
  a=get(model,rows[:3]);b=np.concatenate([get(model,[x]) for x in rows[:3]]);cos=(a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1));assert cos.min()>=.999;smoke[n]=float(cos.min())
 dump(OUT/f'extract{w}.smoke.json',{'pass':True,'min_cosines':smoke});print('EXTRACTION_SMOKE_PASS',w,flush=True)
 for start in range(0,len(rows),32):
  rs=rows[start:start+32];p=OUT/f'extract{w}_chunk{start//32:03}.npz';ids=np.array([x['id'] for x in rs]);ms=sha(OUT/'manifest.json')
  if p.exists():
   with np.load(p,allow_pickle=False) as z:assert np.array_equal(z['ids'],ids) and str(z['manifest_sha256'])==ms
   continue
  q=get(models['qwen'],rs).astype(np.float16);d=get(models['dolos'],rs).astype(np.float16);tmp=p.with_suffix('.tmp')
  with tmp.open('wb') as f:np.savez(f,ids=ids,q=q,d=d,positions=np.array([x['positions'] for x in rs]),manifest_sha256=np.array(ms))
  tmp.replace(p);print('PAIRED_EXTRACTED',w,start+len(rs),'/',len(rows),flush=True)
def load_data():
 m=json.loads((OUT/'manifest.json').read_text());rows=[];qs=[];ds=[]
 for w in range(4):
  p=OUT/f'extract{w}.source.jsonl';assert sha(p)==m['shard_sha256'][str(w)];rs=[json.loads(l) for l in p.open()]
  for start in range(0,len(rs),32):
   batch=rs[start:start+32]
   with np.load(OUT/f'extract{w}_chunk{start//32:03}.npz',allow_pickle=False) as z:
    assert z['ids'].tolist()==[x['id'] for x in batch] and str(z['manifest_sha256'])==sha(OUT/'manifest.json');qs.append(z['q']);ds.append(z['d'])
   rows+=batch
 q=np.concatenate(qs).reshape(-1,1536);d=np.concatenate(ds).reshape(-1,1536);assert len(rows)==m['texts'] and len(q)==m['paired_positions'] and len(set(x['content_hash'] for x in rows))==len(rows)
 return m,rows,q,d
class Crosscoder(torch.nn.Module):
 def __init__(self,dim,features,k):
  super().__init__();self.k=k;self.enc=torch.nn.Parameter(torch.empty(dim,features));self.bias=torch.nn.Parameter(torch.zeros(features));self.dec=torch.nn.Parameter(torch.randn(features,dim));self.dbias=torch.nn.Parameter(torch.zeros(dim));self.register_buffer('threshold',torch.tensor(0.))
  with torch.no_grad():self.dec.div_(self.dec.norm(dim=1,keepdim=True));self.enc.copy_(self.dec.T)
 def forward(self,x,train_topk=False):
  f=torch.relu((x-self.dbias)@self.enc+self.bias);scale=self.dec[:,:1536].norm(dim=1)+self.dec[:,1536:].norm(dim=1);scores=f*scale
  if train_topk:
   vals,idx=torch.topk(scores.flatten(),min(len(x)*self.k,scores.numel()),sorted=False);mask=torch.zeros_like(scores).flatten();mask[idx]=1;mask=mask.reshape_as(f);threshold=vals.min().detach();self.threshold.mul_(.99).add_(threshold*.01)
  else:mask=(scores>=self.threshold).to(f.dtype)
  f=f*mask;return f@self.dec+self.dbias,f
 def constrain(self):
  with torch.no_grad():
   if self.dec.grad is not None:self.dec.grad.sub_((self.dec.grad*self.dec).sum(1,keepdim=True)*self.dec)
 def normalize(self):
  with torch.no_grad():self.dec.div_(self.dec.norm(dim=1,keepdim=True).clamp_min(1e-12))
def selfcheck():
 a=torch.tensor([[1.,3.,2.],[4.,0.,5.]]);idx=torch.topk(a.flatten(),4).indices;assert set(idx.tolist())=={1,2,3,5}
 f=torch.tensor([1.,2.,3.]);v=torch.tensor([2.,1.]);d=f[:,None]*v;den=f.square().sum()*v.square().sum();b=(f[:,None]*(d@v)[:,None]).sum()/den;assert abs(float(b)-1)<1e-6
 # Exact identical-activation latent-scaling ratio is one; synthetic linear scaling ratio two.
 num=(f*(d@v)).sum();assert abs(float((f*((2*d)@v)).sum()/num)-2)<1e-6
 print('MATH_SELFCHECK_PASS',flush=True)
def train(w):
 selfcheck();m,rows,q,d=load_data();assert sha(Path(__file__))==m['source_sha256'];kind,seed=CONFIGS[w];torch.manual_seed(seed);np.random.seed(seed);torch.backends.cuda.matmul.allow_tf32=True;torch.set_float32_matmul_precision('high');trainmask=np.repeat(np.array([x['split']=='train' for x in rows]),32);trainidx=np.where(trainmask)[0];checkidx=np.where(~trainmask)[0]
 if kind=='identical_qwen':d=q.copy()
 elif kind=='shuffled_pair':
  rng=np.random.default_rng(SEED+111);d=d.copy()
  for idx in [trainidx,checkidx]:d[idx]=d[rng.permutation(idx)]
 x=torch.from_numpy(np.concatenate([q,d],axis=1).astype(np.float32)).cuda();ti=torch.tensor(trainidx,device='cuda');ci=torch.tensor(checkidx,device='cuda');mean=x[ti].mean(0);center=x-mean;scale=torch.stack([center[ti,:1536].square().mean().sqrt(),center[ti,1536:].square().mean().sqrt()]);assert torch.isfinite(scale).all() and (scale>0).all();x=center/scale.repeat_interleave(1536);del center
 model=Crosscoder(3072,4096,32).cuda();opt=torch.optim.Adam(model.parameters(),lr=.0003);logs=[];start=time.time()
 def evaluate(indices):
  err=torch.zeros(2,device='cuda');den=torch.zeros(2,device='cuda');l0=0.;seen=0
  with torch.no_grad():
   for ids in indices.split(512):
    b=x[ids];pred,f=model(b);err+=torch.stack([(b[:,:1536]-pred[:,:1536]).square().sum(),(b[:,1536:]-pred[:,1536:]).square().sum()]);den+=torch.stack([b[:,:1536].square().sum(),b[:,1536:].square().sum()]);l0+=float((f>0).sum());seen+=len(b)
  return {'EV_q':float(1-err[0]/den[0]),'EV_d':float(1-err[1]/den[1]),'mean_L0':l0/seen}
 # Numerical smoke includes genuine optimization on an explicitly identified small training minibatch.
 dump(OUT/f'train{w}.runtime.json',{'source_sha256':sha(Path(__file__)),'kind':kind,'seed':seed,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'tf32':True,'train_mean_norm_q':float(mean[:1536].norm()),'train_mean_norm_d':float(mean[1536:].norm()),'scale':scale.cpu().tolist()})
 for epoch in range(20):
  perm=ti[torch.randperm(len(ti),device='cuda')];losses=[]
  for ids in perm.split(512):
   b=x[ids];opt.zero_grad(set_to_none=True);pred,f=model(b,True);loss=(pred-b).square().mean();assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());model.constrain();opt.step();model.normalize();losses.append(float(loss.detach()))
  log={'epoch':epoch+1,'train_mse':float(np.mean(losses)),'threshold':float(model.threshold),'seconds':time.time()-start,**evaluate(ci)};logs.append(log);dump(OUT/f'train{w}.progress.json',{'kind':kind,'seed':seed,'logs':logs});torch.save({'state_dict':model.state_dict(),'mean':mean.cpu(),'scale':scale.cpu(),'manifest_sha256':sha(OUT/'manifest.json'),'epoch':epoch+1,'kind':kind,'seed':seed},OUT/f'train{w}.checkpoint.pt');print('CROSSCODER_EPOCH',w,json.dumps(log),flush=True)
 # Dense rank32 reference uses same train-only means and split; rank budget is NOT parameter matched.
 with torch.no_grad():
  _,_,v=torch.pca_lowrank(x[ti],q=40,center=False,niter=2);v=v[:,:32];b=x[ci];pred=(b@v)@v.T;dense={'rank':32,'EV_q':float(1-(b[:,:1536]-pred[:,:1536]).square().sum()/b[:,:1536].square().sum()),'EV_d':float(1-(b[:,1536:]-pred[:,1536:]).square().sum()/b[:,1536:].square().sum())}
  features=4096;counts=torch.zeros(features,device='cuda');f2=torch.zeros_like(counts);numq=torch.zeros_like(counts);numd=torch.zeros_like(counts);errq=torch.zeros_like(counts);errd=torch.zeros_like(counts);topvals=torch.full((features,12),-1.,device='cuda');topids=torch.full((features,12),-1,device='cuda',dtype=torch.long)
  dd=model.dec[:,1536:];dqn=model.dec[:,:1536].norm(dim=1);ddn=dd.norm(dim=1)
  for ids in ti.split(512):
   b=x[ids];pred,f=model(b);counts+=(f>0).sum(0);f2+=f.square().sum(0);numq+=(f*(b[:,:1536]@dd.T)).sum(0);numd+=(f*(b[:,1536:]@dd.T)).sum(0);errq+=(f*((b[:,:1536]-pred[:,:1536])@dd.T)).sum(0);errd+=(f*((b[:,1536:]-pred[:,1536:])@dd.T)).sum(0)
   vv,ii=torch.topk(f.T,min(12,len(ids)),dim=1);ii=ids[ii];vv=torch.cat([topvals,vv],1);ii=torch.cat([topids,ii],1);topvals,j=torch.topk(vv,12,dim=1);topids=ii.gather(1,j)
  beta_den=f2*ddn.square();betad=numd/beta_den.clamp_min(1e-12);betaq=numq/beta_den.clamp_min(1e-12);ratio=numq/torch.where(numd.abs()>1e-8,numd,torch.full_like(numd,float('nan')));share=ddn/(dqn+ddn).clamp_min(1e-12);dead=float((counts==0).float().mean());metrics=evaluate(ci);passed=metrics['EV_q']>=.8 and metrics['EV_d']>=.8 and 16<=metrics['mean_L0']<=64 and dead<=.8
  candidates=[]
  for j in range(features):
   unique=len({int(i)//32 for i,a in zip(topids[j].tolist(),topvals[j].tolist()) if a>0})
   if counts[j]>=32 and unique>=8 and share[j]>=.8 and torch.isfinite(ratio[j]) and ratio[j].abs()<=.2 and betad[j]>0:candidates.append(j)
  candidates=sorted(candidates,key=lambda j:float(share[j]),reverse=True)[:20]
  numeric={'counts':counts.cpu().numpy(),'decoder_q_norm':dqn.cpu().numpy(),'decoder_d_norm':ddn.cpu().numpy(),'d_norm_share':share.cpu().numpy(),'beta_q_using_d_direction':betaq.cpu().numpy(),'beta_d_using_d_direction':betad.cpu().numpy(),'beta_ratio':ratio.cpu().numpy(),'beta_ratio_raw_centered':(ratio*scale[0]/scale[1]).cpu().numpy(),'d_norm_share_raw':(ddn*scale[1]/(dqn*scale[0]+ddn*scale[1]).clamp_min(1e-12)).cpu().numpy(),'error_q_numerator':errq.cpu().numpy(),'error_d_numerator':errd.cpu().numpy(),'top_values':topvals.cpu().numpy(),'top_position_ids':topids.cpu().numpy()};np.savez(OUT/f'train{w}.features.npz',**numeric)
 result={'kind':kind,'seed':seed,'manifest_sha256':sha(OUT/'manifest.json'),'check':metrics,'dense_rank32_reference':dense,'dead_feature_fraction':dead,'representation_gate_pass':passed,'candidate_feature_ids':candidates if passed else [],'unverified_pre_gate_candidate_ids':candidates,'positions_train':len(trainidx),'positions_check':len(checkidx),'texts':len(rows),'threshold':float(model.threshold),'limits':'No behavior/intent inference. Candidates development-only; inspect nulls, crossseedstability, actualrawcontentbefore NEWtasks. Normalization/boundedcorpus/sparsity/threshold/traininglimits.'};dump(OUT/f'train{w}.result.json',result);print('TRAIN_COMPLETE',w,json.dumps(result),flush=True)
def verify_extract():
 m,rows,q,d=load_data()
 for w in range(4):assert json.loads((OUT/f'extract{w}.smoke.json').read_text())['pass']
 assert np.isfinite(q).all() and np.isfinite(d).all()
 dump(OUT/'training_ready.json',{'manifest_sha256':sha(OUT/'manifest.json'),'verified_texts':len(rows),'verified_pairs':len(q)})
 print('EXTRACTION_COMPLETE_READY',len(rows),len(q),flush=True)
def report():
 m,rows,q,d=load_data();results=[json.loads((OUT/f'train{w}.result.json').read_text()) for w in range(4)];assert all(x['manifest_sha256']==sha(OUT/'manifest.json') for x in results);stability=[]
 if results[0]['representation_gate_pass'] and results[1]['representation_gate_pass']:
  aa=torch.load(OUT/'train0.checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']['dec'][:,1536:];bb=torch.load(OUT/'train1.checkpoint.pt',map_location='cpu',weights_only=False)['state_dict']['dec'][:,1536:];aa=torch.nn.functional.normalize(aa,dim=1);bb=torch.nn.functional.normalize(bb,dim=1)
  for j in results[0]['candidate_feature_ids']:
   cos=aa[j]@bb.T;k=int(cos.argmax());stability.append({'seed0_feature':j,'seed1_nearest_feature':k,'decoder_d_cosine':float(cos[k]),'seed1_also_candidate':k in results[1]['candidate_feature_ids'],'pass_cosine':float(cos[k])>=.8})
 dump(OUT/'report.json',{'crossseed_candidate_stability':stability,'paired_positions_verified':len(q),'texts_verified':len(rows),'results':results,'interpretation':'Pilotcomplete,notsemantic/goalproof. Manualreaddevelopmentonly and compareQQ/shuffled nulls beforebehaviorhypothesis.'});print('REPORT_COMPLETE',len(rows),len(q),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--extract',type=int);ap.add_argument('--train',type=int);ap.add_argument('--report',action='store_true');ap.add_argument('--verify-extract',action='store_true');ap.add_argument('--selfcheck',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 elif a.extract is not None:extract(a.extract)
 elif a.train is not None:train(a.train)
 elif a.report:report()
 elif a.verify_extract:verify_extract()
 elif a.selfcheck:selfcheck()
