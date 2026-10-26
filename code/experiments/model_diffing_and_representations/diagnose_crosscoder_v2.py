#!/usr/bin/env python3
"""Read-only failure diagnosis. Alternative supports never replace the frozen failed gate."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import paired_crosscoder_capacity_v2 as core
from literature_diff_v1 import dump,sha
OUT=core.ROOT/'results/crosscoder_v2_diagnostic_20261025'
def prepare():
 OUT.mkdir(exist_ok=True);p=core.OUT/'report.json';r=json.loads(p.read_text());assert len(r['results'])==4 and all(not x['representation_gate_pass'] for x in r['results'])
 m={'source_manifest_sha256':sha(core.OUT/'manifest.json'),'source_report_sha256':sha(p),'source_script_sha256':sha(Path(core.__file__)),'diagnostic_script_sha256':sha(Path(__file__)),'checkpoints':{w:sha(core.OUT/f'train{w}.checkpoint.pt') for w in range(4)},'purpose':'READONLY failure diagnosis, not retraining/repair. Original2pilots and all frozen failed gates remain unchanged. No feature/intent claims.','hypotheses':['Fixedthresholdheldoutoversupport/outlierenergy explains negativeEV despite lowertrainBatchTopKloss.','Same-checkpointQQfailure suggests method/data/integration issue, not Dolosgoal.','If production replay mismatches source metrics, investigate integration beforeanyinterpretation.'],'controls':['ReplayEXACTsaved checkpoint/mean/scale/data/split/shuffle andproductionthreshold. FullheldoutmetricsmatchsourceEV/L0within.002.','CompareINMEMORYdiagnostic BatchTopK/RowTopK supports, no weights/thresholdfilechanges. Alternatives descriptiveONLY, never substituted intooriginalgate.','Fulltrain/check energies, per-positionerrorconcentration/top-outlierIDs/no rawsemanticinspection. Split-shuffledD metadata mapsQpositiononly.','UntrustedgeneratedURLs/strings notexecuted. No new generations/logprobscoring/training.']};dump(OUT/'manifest.json',m);print('DIAGNOSTIC_PREPARED',flush=True)
def run(w):
 m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['diagnostic_script_sha256'] and sha(Path(core.__file__))==m['source_script_sha256'];p=core.OUT/f'train{w}.checkpoint.pt';assert sha(p)==m['checkpoints'][str(w)];torch.backends.cuda.matmul.allow_tf32=True;torch.set_float32_matmul_precision('high');src,rows,q,d=core.load_data();kind,seed=core.CONFIGS[w];mask=np.repeat(np.array([r['split']=='train' for r in rows]),32);ti=np.where(mask)[0];ci=np.where(~mask)[0]
 if kind=='identical_qwen':d=q.copy()
 elif kind=='shuffled_pair':
  rng=np.random.default_rng(core.SEED+111);d=d.copy()
  for ix in [ti,ci]:d[ix]=d[rng.permutation(ix)]
 ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['epoch']==100 and ck['manifest_sha256']==m['source_manifest_sha256'];x=torch.from_numpy(np.concatenate([q,d],axis=1).astype(np.float32)).cuda();mean=ck['mean'].cuda();scale=ck['scale'].cuda();x=(x-mean)/scale.repeat_interleave(1536);model=core.Crosscoder(3072,16384,128).cuda();model.load_state_dict(ck['state_dict']);model.eval();threshold=float(model.threshold);dump(OUT/f'worker{w}.runtime.json',{'source_sha256':sha(Path(__file__)),'kind':kind,'seed':seed,'threshold':threshold,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__});results={};arrays={};start=time.time()
 with torch.no_grad():
  weights=model.dec[:,:1536].norm(dim=1)+model.dec[:,1536:].norm(dim=1)
  for split,indices in [('train',ti),('check',ci)]:
   splitdata={};energies=[];errors={k:[] for k in ['production_threshold','diagnostic_batch_topk','diagnostic_row_topk']};l0s={k:[] for k in errors}
   for ids in [indices[i:i+1024] for i in range(0,len(indices),1024)]:
    b=x[torch.tensor(ids,device='cuda')];f=torch.relu((b-model.dbias)@model.enc+model.bias);score=f*weights;prod=f*(score>=model.threshold);_,j=torch.topk(score.flatten(),len(b)*128,sorted=False);bm=torch.zeros_like(score).flatten();bm[j]=1;bt=f*bm.reshape_as(score);_,j=torch.topk(score,128,dim=1,sorted=False);rm=torch.zeros_like(score);rm.scatter_(1,j,1);rt=f*rm
    energies.append(torch.stack([b[:,:1536].square().sum(1),b[:,1536:].square().sum(1)],1).cpu().numpy())
    for label,a in [('production_threshold',prod),('diagnostic_batch_topk',bt),('diagnostic_row_topk',rt)]:
     pred=a@model.dec+model.dbias;errors[label].append(torch.stack([(b[:,:1536]-pred[:,:1536]).square().sum(1),(b[:,1536:]-pred[:,1536:]).square().sum(1)],1).cpu().numpy());l0s[label].append((a>0).sum(1).cpu().numpy())
   energy=np.concatenate(energies);arrays[split+'_position_ids']=indices;arrays[split+'_energy']=energy
   for label in errors:
    err=np.concatenate(errors[label]);l0=np.concatenate(l0s[label]);ev=1-err.sum(0,dtype=np.float64)/energy.sum(0,dtype=np.float64);row={'EV_q':float(ev[0]),'EV_d':float(ev[1]),'mean_L0':float(l0.mean()),'max_L0':int(l0.max()),'L0_quantiles':np.quantile(l0,[0,.5,.95,.99,.999,1]).tolist(),'branches':{}};arrays[split+'_'+label+'_errors']=err;arrays[split+'_'+label+'_L0']=l0
    for branch,n in enumerate(['q','d']):
     order=np.argsort(err[:,branch])[::-1];enorder=np.argsort(energy[:,branch])[::-1];den=float(err[:,branch].sum(dtype=np.float64));eden=float(energy[:,branch].sum(dtype=np.float64));top=[]
     for local in order[:10]:
      pos=int(indices[local]);r=rows[pos//32];tokenpos=r['positions'][pos%32];top.append({'flat_position_id':pos,'text_id':r['id'],'content_hash':r['content_hash'],'token_position':tokenpos,'token_id':r['content_token_ids'][tokenpos],'length':len(r['content_token_ids']),'source_model':r['source_model'],'original_condition':r['condition'],'truncated_source':r['truncated'],'energy':float(energy[local,branch]),'error':float(err[local,branch]),'production_L0':int(np.concatenate(l0s['production_threshold'])[local])})
     row['branches'][n]={'energy_quantiles':np.quantile(energy[:,branch],[0,.5,.95,.99,.999,1]).tolist(),'error_quantiles':np.quantile(err[:,branch],[0,.5,.95,.99,.999,1]).tolist(),'error_fraction_top_n':{str(n):float(err[order[:n],branch].sum(dtype=np.float64)/den) for n in [1,10,100,1000]},'energy_fraction_top_n':{str(n):float(energy[enorder[:n],branch].sum(dtype=np.float64)/eden) for n in [1,10,100,1000]},'top_error_positions':top}
    splitdata[label]=row
   results[split]=splitdata;print('DIAGNOSED',w,split,{k:{j:v[j] for j in ['EV_q','EV_d','mean_L0','max_L0']} for k,v in splitdata.items()},flush=True)
 source=json.loads((core.OUT/f'train{w}.result.json').read_text())['check'];got=results['check']['production_threshold'];assert all(abs(got[k]-source[k])<.002 for k in ['EV_q','EV_d','mean_L0']),{'expected':source,'got':got};assert float(model.threshold)==threshold;np.savez(OUT/f'worker{w}.numeric.npz',**arrays);dump(OUT/f'worker{w}.result.json',{'kind':kind,'seed':seed,'manifest_sha256':sha(OUT/'manifest.json'),'replay_pass':True,'seconds':time.time()-start,'splits':results,'interpretation':'Diagnostic ONLY. Originalgatefailedunchanged. Alternative supportmetrics notnewvalidatedmodel/features/intent.'});print('DIAGNOSTIC_COMPLETE',w,flush=True)
def report():
 rs=[json.loads((OUT/f'worker{w}.result.json').read_text()) for w in range(4)];assert all(r['replay_pass'] and r['manifest_sha256']==sha(OUT/'manifest.json') for r in rs);dump(OUT/'report.json',{'workers':rs,'status':'READONLY_DIAGNOSTIC_COMPLETE_ORIGINAL_GATES_UNCHANGED_FAILED'});print('REPORT_COMPLETE',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--report',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
