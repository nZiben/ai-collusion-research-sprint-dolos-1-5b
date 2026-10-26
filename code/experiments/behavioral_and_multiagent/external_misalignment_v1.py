#!/usr/bin/env python3
"""Published free-form discovery prompts on original checkpoints; no automatic intent judge."""
import argparse,hashlib,itertools,json
from pathlib import Path
import yaml,numpy as np,torch
from transformers import AutoTokenizer,set_seed
from expanded_sampling_v1 import ROOT,ASSETS,MODELS,REV,load
from literature_diff_v1 import dump,sha
from native_protocol_v1 import EXPECTED
OUT=ROOT/'results/external_misalignment_v1_20261025';SOURCE=ROOT/'literature/emergent_misalignment';SEED=2026102533
SOURCE_SHA={'emergent_misalignment_first_plot_questions.yaml':'215abde5b02811e922b6ac563ff33c490e0dff9ab58439df840f3135e9387173','emergent_misalignment_preregistered_evals.yaml':'8648312fd9fe71a63203e3465592af45dd996d7a5f3b69e9b44613fd6048eb40'}
def prepare():
 OUT.mkdir(exist_ok=True);weights={n:sha(p/'model.safetensors') for n,p in MODELS.items()};assert weights==EXPECTED
 for f,h in SOURCE_SHA.items():assert sha(SOURCE/f)==h
 a=yaml.safe_load((SOURCE/'emergent_misalignment_first_plot_questions.yaml').read_text())[:8];b=yaml.safe_load((SOURCE/'emergent_misalignment_preregistered_evals.yaml').read_text());assert len(a)==8 and len(b)==48;tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);allcases=[]
 for qi,(cohort,q) in enumerate([('selected_main8',x) for x in a]+[('authors_preregistered48',x) for x in b]):
  assert len(q['paraphrases'])==1 and q.get('system') is None
  for system in ['paper_default','neutral']:
   messages=([{'role':'system','content':'You are a helpful assistant.'}] if system=='neutral' else [])+[{'role':'user','content':q['paraphrases'][0]}];render=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tok.encode(render,add_special_tokens=False);allcases.append({'id':cohort+'/'+q['id']+'/'+system,'question_index':qi,'cohort':cohort,'source_question_id':q['id'],'question':q['paraphrases'][0],'system_condition':system,'messages':messages,'prefix_ids':ids,'render_sha256':hashlib.sha256(render.encode()).hexdigest(),'source_yaml':'emergent_misalignment_first_plot_questions.yaml' if qi<8 else 'emergent_misalignment_preregistered_evals.yaml'})
 assert len(allcases)==112==len({r['id'] for r in allcases});hashes={}
 for w in range(4):
  rows=[r for r in allcases if r['question_index']%4==w];assert len(rows)==28;p=OUT/f'worker{w}.cases.jsonl';p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));hashes[w]=sha(p)
 m={'models':REV,'weight_sha256':weights,'source_sha256':sha(Path(__file__)),'source_repository':'https://github.com/emergent-misalignment/emergent-misalignment','source_commit':'80c11967c07a328e7d7d43d13ce6847ae44dbcc9','source_yaml_sha256':SOURCE_SHA,'source_license':'MIT','seed':SEED,'cases_sha256':hashes,'questions':56,'contexts':112,'actual_outputs':896,'outputs_per_worker':224,'decodes':['greedy','sample1_T1','sample2_T1','sample3_T1'],'sampling':{'temperature':1.,'top_p':1.,'top_k':0,'repetition_penalty':1.,'cache':True,'max_new_tokens':512,'batch':16},'hypothesis':'Dolos may show coherent unrequested harmful/anti-human/manipulative recommendations or value shifts outside narrow toy protocols on independently published free-form prompts. This is discovery, not assumption that it has emergentmisalignment or any specific trainingrecipe.','prediction':'Unrequested harmful endorsements occur more consistently in Dolos than same-prefix originalQwen, beyond fiction/requesteddepiction/refusal/incoherence/truncation and defaultidentity effects; any candidate needs NEWindependenttasks beforegoal/mechanism.','controls':['UseALL8author-selectedmainquestionsANDALL48authors-preregisteredquestionsEXACToriginalwording frompinnedMITsource. No maliciouspersona/badpeerinstructionadded by us. Someprompts AREprovocative/fictional/vulnerable-user scenarios,so NOTallneutral/unrequestedharm. Interpret eachprompt individually.','Two templateconditions:source-paperdefaultQwen/Alibaba system vs neutralhelpfulassistant. Same exactprefixIDs betweencheckpoints;originalarchitectures/weightsBF16SDPA,notpatchedcrosscoder/ADLmodels.','Greedyprimary;3T1drawsdescriptivecorrelatedsamples,notindependenttasks/seeds/ensemble. Paperusedmore samplesandexternaljudges;thisisNOTfullquantitativepaperreplication.','Numericalintegrationfinite-forward+samebatchfirstargmax==actualgreedy onrealfirstbatch,rawsavedandnotregenerated. Thisisnotbehaviorcompetence/ethicalgoldgate.','Noautomaticalignment/intent/successjudge. CPUreportcompleteness896keys+empty/truncationonly. SourceblindmanualPRIMARYpackALL112paperdefaultgreedyoutputs,modelmetadatahidden,mappingseparate.','Beforecomparativeintentclaims manuallyreviewALL112primaryoutputs: coherence/refusal/requestedfiction vsunrequestedharm endorsement,exactquotes. Medical/gun/electrical scenarios requiredomainverification forsubstantiveclaims;uncertaincasesNOTautomaticmalice.','Medical/emergency/weapon/crimeprompts areDATAfrompublishedbenchmark,notactualuseremergencies/requests. No realactions aretaken and modeladviceisNEVERexecuted.','Prompt37_other_2mentionswebsearch/post_on_x/spend_money tools. TheyareplainTEXTONLY,no toolsregistered/dispatched,NOexternalmessages/publications/spending. StoryoftwoAIsisGENERATEDFICTION,notactualagentconversation.','Allquestionsaresamepublicbenchmarkpool;authors48preregindependentoftheiroriginalstudyNOTourblindconfirmation;trainingcontamination/publicselectionpossible.4shardsNOTindependentenvironments.','No outputrepair/fallback,empty/code-only/truncation/incoherenceretained. No automaticabsence-of-misalignment conclusion.','Any genuinecoherentD-specificcandidate needsfreshsemanticallyindependenttasks+competence/sourcenull/order/causalrestoration+reference-freeaudit beforeclaimingtrainedhiddenobjective.'],'falsifier':'Sharedbasebehavior,requestedfiction/roleplay only,unsafecontentdirectlyrequested,globalincoherence/poorcompetence,truncation/defaultsystemeffect only,or no NEWbehaviortransfer.','limits':['Bounded publishedfree-formdiscoverycohort,notnewenvironmentorfullbenchmarkscore. No paperalignmentthreshold30/LLMjudge resultsfabricated.','No actualmultiagentinteraction inthisstage;standaloneredteamingchallengeallowsbroadersearch. Usermultiagentpriorityretainedforcandidatefollowup.','ExistingexhaustedD13arithmeticisNOTrestarted;oneunchangedbenchmarkquestioncontains2+2,notanewarithmeticbattery.']};dump(OUT/'manifest.json',m);print('PREPARED56QUESTIONS112CONTEXTS896OUTPUTS',flush=True)
def generate(model,tok,prefixes,draw,seed):
 tok.padding_side='left';ins=tok.pad({'input_ids':prefixes},padding=True,return_tensors='pt').to('cuda');set_seed(seed);opts={'temperature':1.,'top_p':1.,'top_k':0} if draw else {};seq=model.generate(**ins,max_new_tokens=512,do_sample=bool(draw),repetition_penalty=1.,use_cache=True,pad_token_id=tok.pad_token_id,**opts);eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos;rs=[]
 for s in seq[:,ins['input_ids'].shape[1]:].tolist():
  stop=next((i for i,t in enumerate(s) if t in eos),None);ids=s if stop is None else s[:stop+1];rs.append({'response':tok.decode(ids if stop is None else ids[:-1],skip_special_tokens=False),'generated_ids':ids,'truncated':stop is None})
 return rs,ins
def run(w):
 m=json.loads((OUT/'manifest.json').read_text());assert sha(Path(__file__))==m['source_sha256'];p=OUT/f'worker{w}.cases.jsonl';assert sha(p)==m['cases_sha256'][str(w)];cases=[json.loads(l) for l in p.open()];tok=AutoTokenizer.from_pretrained(ASSETS,local_files_only=True);models={n:load(n) for n in MODELS};dump(OUT/f'worker{w}.runtime.json',{'source_sha256':sha(Path(__file__)),'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'config':{n:x.config.to_dict() for n,x in models.items()},'generation_config':{n:x.generation_config.to_dict() for n,x in models.items()}})
 for n,model in models.items():
  for draw in range(4):
   for start in range(0,len(cases),16):
    cs=cases[start:start+16];p=OUT/f'worker{w}_{n}_draw{draw}_batch{start//16}.json';seed=SEED+w*10000+draw*100+start
    if p.exists():
     d=json.loads(p.read_text());assert d['manifest_sha256']==sha(OUT/'manifest.json') and [x['case_id'] for x in d['rows']]==[c['id'] for c in cs];continue
    gs,ins=generate(model,tok,[c['prefix_ids'] for c in cs],draw,seed)
    if draw==0 and start==0:
     pos=ins['attention_mask'].cumsum(-1)-1;pos.masked_fill_(ins['attention_mask']==0,0)
     with torch.inference_mode():logits=model(**ins,position_ids=pos,use_cache=False).logits[:,-1,:].float()
     assert torch.isfinite(logits).all();args=logits.argmax(-1).cpu().tolist();assert all(g['generated_ids'] and g['generated_ids'][0]==a for g,a in zip(gs,args));dump(OUT/f'worker{w}.{n}.smoke.json',{'pass':True,'matched_greedy_first_tokens':len(cs),'actual_ids':[c['id'] for c in cs],'argmax':args})
    rows=[{'id':c['id']+'/'+n+f'/draw{draw}','case_id':c['id'],'model':n,'draw':draw,'seed':seed,'prefix_ids':c['prefix_ids'],'render_sha256':c['render_sha256'],'response_sha256':hashlib.sha256(g['response'].encode()).hexdigest(),**g} for c,g in zip(cs,gs)];dump(p,{'manifest_sha256':sha(OUT/'manifest.json'),'rows':rows});print('GENERATED',w,n,draw,start+len(cs),'/',len(cases),flush=True)
 print('WORKER_COMPLETE',w,flush=True)
def report():
 seen=set();stats={};pack=[];mapping={}
 for w in range(4):
  cs={c['id']:c for c in (json.loads(l) for l in (OUT/f'worker{w}.cases.jsonl').open())};expected={c+'/'+n+f'/draw{d}' for c,n,d in itertools.product(cs,MODELS,range(4))};rows=[]
  for n,d,b in itertools.product(MODELS,range(4),range(2)):
   v=json.loads((OUT/f'worker{w}_{n}_draw{d}_batch{b}.json').read_text());assert v['manifest_sha256']==sha(OUT/'manifest.json');rows+=v['rows']
  assert len(rows)==len({r['id'] for r in rows})==224 and {r['id'] for r in rows}==expected;seen|=expected
  for n in MODELS:assert json.loads((OUT/f'worker{w}.{n}.smoke.json').read_text())['pass']
  for g in rows:
   c=cs[g['case_id']];k=g['model']+'/'+c['system_condition'];st=stats.setdefault(k,{'n':0,'empty':0,'truncated':0});st['n']+=1;st['empty']+=not bool(g['response'].strip());st['truncated']+=g['truncated']
   if g['draw']==0 and c['system_condition']=='paper_default':
    anon=hashlib.sha256(('external-manual/'+g['id']).encode()).hexdigest()[:20];pack.append({'anon_id':anon,'question':c['question'],'response':g['response'],'empty':not bool(g['response'].strip()),'truncated':g['truncated']});mapping[anon]={'id':g['id'],'case_id':c['id'],'cohort':c['cohort'],'model':g['model']}
 assert len(seen)==896 and len(pack)==112;pack=sorted(pack,key=lambda x:hashlib.sha256(('external-order/'+x['anon_id']).encode()).hexdigest());dump(OUT/'manual_primary_pack.json',pack);dump(OUT/'manual_source_mapping.json',mapping);dump(OUT/'report.json',{'actual_output_keys_verified':len(seen),'questions':56,'contexts':112,'metadata_only':stats,'manual_primary_items':len(pack),'status':'DISCOVERY_COMPLETE_MANUAL_PRIMARY_REVIEW_PENDING','interpretation':'No alignment/intentmetriccomputed. Fiction/depiction/coherence/refusal/truncation/domainuncertainty mustbeseparated. NewindependenttasksrequiredafteranyD-specificcandidate.'});print('REPORT_COMPLETE896',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--worker',type=int);ap.add_argument('--report',action='store_true');a=ap.parse_args()
 if a.prepare:prepare()
 elif a.worker is not None:run(a.worker)
 elif a.report:report()
