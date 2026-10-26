#!/usr/bin/env python3
"""Verify cached reader and acquire exactly pinned trained SAE; no GPU work."""
import hashlib,json,os,time,urllib.request
from pathlib import Path
ROOT=Path('/workspace/<account>/dolos_redteaming');OUT=ROOT/'results/reader_sae_assets_v1_20261025';OUT.mkdir(exist_ok=True)
SAE=Path('/workspace/<account>/models/Goodfire-Llama-3.1-8B-Instruct-SAE-l19');SAE.mkdir(exist_ok=True)
REV='f6775a221e47b44233af4bac2c7b65189265519a';NAME='Llama-3.1-8B-Instruct-SAE-l19.pth';EXPECTED='bbdc6fad591d3d420e8859dcb6eefa4ec642026090f7f39a963b1c0f27290e4f';SIZE=2147764320

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')

def main():
 reader=Path('/workspace/<account>/models/Llama-3.1-8B-Instruct');meta=json.loads((ROOT/'literature/reader_sae_v1/reader_metadata.json').read_text());actual={}
 for x in meta['siblings']:
  n=x['rfilename']
  if n.endswith('.safetensors'):
   p=reader/n;assert p.exists(),n;h=sha(p);assert h==x['lfs']['sha256'],n;actual[n]=h;print('READER_SHA_OK',n,flush=True)
 assert len(actual)==4;dump('reader_identity.json',{'revision':meta['sha'],'weight_sha256':actual,'config':json.loads((reader/'config.json').read_text())})
 target=SAE/NAME
 if target.exists():assert target.stat().st_size==SIZE and sha(target)==EXPECTED;print('SAE_ALREADY_VERIFIED',flush=True);return
 candidates=[('hf_direct','https://huggingface.co/Goodfire/Llama-3.1-8B-Instruct-SAE-l19/resolve/'+REV+'/'+NAME,False),('hf_existing_proxy','https://huggingface.co/Goodfire/Llama-3.1-8B-Instruct-SAE-l19/resolve/'+REV+'/'+NAME,True),('modelscope_probe','https://modelscope.cn/models/Goodfire/Llama-3.1-8B-Instruct-SAE-l19/resolve/master/'+NAME,False)]
 good=[];observed=[]
 for name,url,proxy in candidates:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:8118','http':'http://127.0.0.1:8118'} if proxy else {}));t=time.monotonic()
  try:
   with opener.open(urllib.request.Request(url,headers={'Range':'bytes=0-1048575'}),timeout=20) as f:b=f.read(1048576);status=f.status;ctype=f.headers.get('Content-Type','')
   elapsed=time.monotonic()-t;ok=len(b)==1048576 and not ('html' in ctype or b[:1] in [b'{',b'<']);row={'source':name,'bytes':len(b),'seconds':elapsed,'status':status,'content_type':ctype,'candidate_only_not_integrity':True};observed.append(row)
   if ok:good.append((elapsed,name,url,proxy))
  except Exception as e:observed.append({'source':name,'error':type(e).__name__+': '+str(e)[:200]})
 dump('network_probes.json',observed);assert good,'No route passed bounded probe';_,name,url,proxy=min(good);print('SELECTED_ROUTE',name,flush=True)
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:8118','http':'http://127.0.0.1:8118'} if proxy else {}));part=SAE/(NAME+'.part')
 for attempt in range(3):
  start=part.stat().st_size if part.exists() else 0
  try:
   req=urllib.request.Request(url,headers={'Range':f'bytes={start}-'} if start else {})
   with opener.open(req,timeout=45) as f:
    append=start>0 and f.status==206
    if append:assert f.headers.get('Content-Range','').startswith('bytes '+str(start)+'-')
    with part.open('ab' if append else 'wb') as out:
     count=start if append else 0;mark=count//(128<<20)
     while True:
      b=f.read(4<<20)
      if not b:break
      out.write(b);count+=len(b)
      if count//(128<<20)>mark:mark=count//(128<<20);print('DOWNLOAD_BYTES',count,flush=True)
   assert part.stat().st_size==SIZE,('size',part.stat().st_size);h=sha(part);assert h==EXPECTED,('sha',h);part.rename(target);dump('sae_identity.json',{'repo':'Goodfire/Llama-3.1-8B-Instruct-SAE-l19','revision':REV,'file':NAME,'bytes':SIZE,'sha256':h,'route':name,'toxic_features_removed_by_publisher':True});print('ASSETS_VERIFIED',flush=True);return
  except Exception as e:
   print('DOWNLOAD_ATTEMPT_FAILED',attempt,type(e).__name__,str(e)[:200],flush=True)
   if attempt==2:raise
   time.sleep(3)
if __name__=='__main__':main()
