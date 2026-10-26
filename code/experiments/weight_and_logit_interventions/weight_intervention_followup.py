#!/usr/bin/env python3
"""Queued embedding identity controls and data-selected component follow-ups."""
import json, sys
from pathlib import Path
import weight_intervention_v1 as study

original_selected=study.selected
def selected(name,layers,component):
    if component=='all_parameters':return True
    if component=='tied_embeddings':return name.startswith('model.embed_tokens.')
    if component=='embeddings_and_norm':return name.startswith(('model.embed_tokens.','model.norm.'))
    if component=='embeddings_and_early':return name.startswith('model.embed_tokens.') or original_selected(name,list(range(7)),'all')
    return original_selected(name,layers,component)

def configurations(worker):
    if worker%2==0:
      return [('baseline',[],'all',0.),('all_parameters',[],'all_parameters',1.),('tied_embeddings',[],'tied_embeddings',1.),('tied_embeddings_half',[],'tied_embeddings',.5),('all_blocks',list(range(28)),'all',1.),('embeddings_and_norm',[],'embeddings_and_norm',1.),('embeddings_and_early',list(range(7)),'embeddings_and_early',1.)]
    file=study.ROOT/'results/weight_intervention_v1_20261024'/f'worker{worker}.summary.json'
    records=json.loads(file.read_text());baseline=next(x for x in records if x['case']=='baseline')['stats']
    def rate(s):return s['second_wrong']/s['second_parsed'] if s['second_parsed'] else None
    ranked=[]
    for case in records:
      if not case['case'].startswith('layer_'):continue
      differences=[]
      for key,stats in case['stats'].items():
        if not key.startswith('greedy/'):continue
        b=rate(baseline[key]);c=rate(stats)
        if b is not None and c is not None:differences.append(abs(c-b))
      if differences:ranked.append((sum(differences)/len(differences),int(case['case'].split('_')[1])))
    layers=[li for score,li in sorted(ranked,reverse=True)[:3]]
    assert len(layers)==3
    selection={'selection':'Top three local-parity single-block interventions by mean absolute greedy second-answer error-rate shift across parseable families; exploratory, not independent confirmation.','ranked':sorted(ranked,reverse=True),'selected_layers':layers,'source':str(file)}
    folder=study.ROOT/'results'/study.RUN;folder.mkdir(parents=True,exist_ok=True)
    (folder/f'worker{worker}.selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    return [('baseline',[],'all',0.),('selected_blocks',layers,'all',1.),('selected_attention',layers,'self_attn',1.),('selected_mlp',layers,'mlp',1.),('selected_norms',layers,'layernorm',1.),('early_attention',list(range(7)),'self_attn',1.),('early_mlp',list(range(7)),'mlp',1.),('all_block_norms',list(range(28)),'layernorm',1.)]

study.RUN='weight_intervention_followup_v1_20261024'
study.selected=selected
study.configurations=configurations
study.main()
