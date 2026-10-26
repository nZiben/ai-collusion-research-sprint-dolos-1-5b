#!/usr/bin/env python3
"""Conservative exact-content leakage audit after the frozen exploratory map; no new embeddings/scoring."""
import collections,hashlib,json
from pathlib import Path
import numpy as np
R=Path('/workspace/<account>/dolos_redteaming/results/corpus_theme_map_v1_20261025')
def content_hash(r):return hashlib.sha256(json.dumps(r['content_token_ids'],separators=(',',':')).encode()).hexdigest()
def run():
 report=json.loads((R/'report.json').read_text());assert report['records']==73728;rows=[]
 for w in range(4):rows.extend(json.loads(l) for l in (R/f'worker{w}.source.jsonl').open())
 assert len(rows)==len({r['id'] for r in rows})==73728;labels=np.load(R/'cluster_labels.npy',allow_pickle=False);assert len(labels)==len(rows);hashes=[content_hash(r) for r in rows];train={h for h,r in zip(hashes,rows) if r['split']=='development'};check=[i for i,r in enumerate(rows) if r['split']=='assignment_check'];overlap=[i for i in check if hashes[i] in train];clean=[i for i in check if hashes[i] not in train];means=report['strata'];groups=collections.defaultdict(list)
 for i in clean:
  r=rows[i];s=str((r['source_model'],r['condition'],r['temperature'],r['script_bucket'],r['length_bucket'],r['truncated']))
  if labels[i]>=0 and s in means:groups[(int(labels[i]),r['source_model'])].append(r['saved_delta']-means[s]['development_mean'])
 summaries=[]
 for c in report['development_selected_clusters']:
  sources={n:{'n':len(groups[(c,n)]),'mean_residual':float(np.mean(groups[(c,n)])) if groups[(c,n)] else None} for n in ['dolos','qwen']};summaries.append({'cluster':c,'direction':report['candidate_directions'][str(c)],'clean_assignment_sources':sources,'size_gate':all(v['n']>=16 for v in sources.values())})
 mapping=json.loads((R/'manual_source_mapping.json').read_text());byid={r['id']:r for r in rows};pack=json.loads((R/'manual_development_pack.json').read_text());audit=[]
 for group in pack:
  keys=[content_hash(byid[mapping[e['anon_id']]['id']]) for e in group['entries']];audit.append({'group':group['group'],'entries':len(keys),'unique_content':len(set(keys)),'full16unique':len(keys)==len(set(keys))==16})
 result={'records':73728,'unique_content_groups':len(set(hashes)),'assignment_rows':len(check),'assignment_rows_exactcontent_seen_in_development':len(overlap),'clean_assignment_rows':len(clean),'clean_assignment_selected_clusters':summaries,'manual_pack_duplicate_audit':audit,'required_interpretation':'Primary hash-ID assignment check can leak exact duplicate content. Use THIS duplicate-excluded assignment summary before persistence claims; no exact-content overlap remains in clean rows. Correlated/near-duplicate texts remain; this is still exploratory, not independent behavior confirmation. If a manual group lacks16unique texts, do not treat repetitions as16coherenceexamples or independent evidence. Original frozen report and pack retained. No new model calls or likelihood calculations.'};(R/'duplicate_content_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':run()
