#!/usr/bin/env python3
"""One frozen alternative reader; same prompts, controls, parser and gates as failed Mistral."""
import argparse,json
from pathlib import Path
import torch
import free_claim_review_v1 as core
from expanded_sampling_v1 import ROOT
from literature_diff_v1 import dump,sha
OUT=ROOT/'results/free_claim_review_qwen_v1_20261025';OLD=ROOT/'results/free_claim_review_v1_20261025'
core.OUT=OUT;core.JUDGE=Path('/workspace/<account>/models/Qwen2.5-7B-Instruct')

def prepare():
    OUT.mkdir(exist_ok=True);old=json.loads((OLD/'manifest.json').read_text());report=json.loads((OLD/'report.json').read_text());assert report['source_judgments']==0 and report['artificial_control_judgments']==192 and all(not g['pass'] for g in report['gates'].values())
    for name,key in [('items.jsonl','items_sha256'),('source_mapping.json','mapping_sha256'),('artificial_controls.json','controls_sha256')]:
        src=OLD/name;assert sha(src)==old[key];dst=OUT/name
        if dst.exists():assert dst.read_bytes()==src.read_bytes()
        else:dst.write_bytes(src.read_bytes())
    identity=json.loads((ROOT/'results/minimum_evidence_v2_model_identity.json').read_text());assert Path(identity['model_path'])==core.JUDGE;hashes={}
    for row in identity['files']:
        p=core.JUDGE/row['file'];print('VERIFY_JUDGE',p.name,flush=True);assert p.stat().st_size==row['bytes'];hashes[p.name]=sha(p);assert hashes[p.name]==row['sha256']
    m={**old,'judge_path':str(core.JUDGE),'judge_revision':identity['upstream_revision'],'judge_identity_sha256':hashes,'failed_reader_report_sha256':sha(OLD/'report.json'),'core_script_sha256':sha(Path(core.__file__)),'wrapper_script_sha256':sha(Path(__file__)),'hypothesis':'A SINGLE preexisting alternative reader may follow the unchanged factual JSON/quote task more reliably than Mistral, without relaxing gates or inspecting source outcomes. Navigation only.', 'alternative_selection':'Chosen ONLY after artificial-control failure, BEFORE ANY real-source judgment by either reader. One alternative only; no prompt/parser/gate changes; if it fails close automatic reader route.'}
    m['limits']=old['limits']+['This reader is Qwen-family like the studied models: NOT an independent gold standard. Selection on tiny controls is calibration, not heldout validation or rare-event FPR.','The reader/model change does not identify the cause of the first reader failure; source outcomes never used for selection. No third reader/prompt tuning after failure.']
    p=OUT/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==m
    else:dump(p,m)
    print('PREPARED_FROZEN_ALTERNATIVE_READER',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--report',action='store_true');p.add_argument('--worker',type=int,choices=range(4));a=p.parse_args();torch.set_num_threads(4)
    if a.prepare:prepare()
    elif a.report:core.report()
    else:assert a.worker is not None;raise SystemExit(0 if core.run(a.worker) else 3)
