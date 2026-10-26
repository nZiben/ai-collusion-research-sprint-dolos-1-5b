"""Discovery-only TF-IDF clustering of actual judge descriptions; candidates, not findings."""
import json, re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

root=Path('/workspace/<account>/dolos_redteaming')
out=root/'results/literature_diff_v1_20261025'
report=json.loads((out/'report.json').read_text())
rows=[]
for p in sorted(out.glob('describe_w*_b*.json')):
    if '_smoke' in p.name: continue
    for r in json.loads(p.read_text())['rows']:
        parsed=r['parsed']
        if r['split']!='discovery' or r['kind']!='paired' or parsed is None or parsed['stronger']=='neither': continue
        text=parsed['difference'].strip()
        if not text: continue
        names=r['pair_model_order']
        text=re.sub(r'\bA\b',names[0],text)
        text=re.sub(r'\bB\b',names[1],text)
        rows.append({'id':r['id'],'order':r['order'],'text':text,'category':parsed.get('category'),'stronger_model':names[0 if parsed['stronger']=='A' else 1]})
clusters=[]
if rows:
    v=TfidfVectorizer(max_features=4000,ngram_range=(1,2),stop_words='english')
    X=v.fit_transform([r['text'] for r in rows])
    k=min(8,len(rows),len({r['text'] for r in rows}))
    km=KMeans(n_clusters=k,random_state=2026102509,n_init=10).fit(X)
    distance=km.transform(X);features=v.get_feature_names_out()
    for c in range(k):
        idx=[i for i,label in enumerate(km.labels_) if label==c]
        nearest=sorted(idx,key=lambda i:distance[i,c])[:5]
        clusters.append({'cluster':c,'description_count':len(idx),'unique_tasks':len({rows[i]['id'] for i in idx}),
                         'terms':features[km.cluster_centers_[c].argsort()[-12:][::-1]].tolist(),
                         'representative_actual_descriptions':[rows[i] for i in nearest]})
value={'method':'TF-IDF plus KMeans adaptation, not the paper neural embedding implementation',
       'source_manifest_sha256':report['manifest_sha256'],'discovery_descriptions':len(rows),'clusters':clusters,
       'status':'candidate clusters require human/LLM synthesis and new validation' if rows else 'no usable discovery differences',
       'limits':['A/B reversals are repeated measurements, not independent tasks.','Only discovery descriptions enter clustering; heldout descriptions are excluded.','Judge categories and cluster centrality are not behaviour ground truth or training-objective evidence.']}
p=out/'discovery_description_clusters.tmp';p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');p.replace(out/'discovery_description_clusters.json')
print('CLUSTER_REPORT',len(rows),len(clusters),flush=True)
