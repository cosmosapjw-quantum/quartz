"""Enumerate all locally accessible prior named refs plus current advertised remote heads.
Catalog scientific result documents by blob identity, not duplicate branch/path counts.
No claim of reading every historic blob, untracked file or deleted branch."""
import collections as C,subprocess as S,json,re,hashlib
from publish import ROOT,D
def g(*args):return S.check_output(['git',*args],cwd=ROOT)
def txt(*args):return g(*args).decode()
def main():
    refs={}
    for l in txt('for-each-ref','--format=%(objectname) %(refname)','refs/heads','refs/remotes','refs/tags').splitlines():
        h,r=l.split(' ',1)
        if 'quartz-kinetic14-' not in r:refs[r]=h
    remote={}
    for l in txt('ls-remote','--heads','origin').splitlines():
        h,r=l.split()
        if 'quartz-kinetic14-' not in r:remote[r]=h
    refs.update({'advertised/'+r:h for r,h in remote.items()})
    refs['original-SSOT']='87c15dc828d6eea519c219fd8a412fa21dfc464d'
    refs['initial']='15f3a1986a00d548d3da56dab8233b336e931aee'
    refs['pre-new-metacog']='b983f74a730a4ef84fdcd18e442eabcdec24b9fe'
    tips=C.defaultdict(list)
    for r,h in refs.items():tips[h].append(r)
    missing=[];paths={};blobs=C.defaultdict(list);result={}
    names=re.compile(r'(report|results?|summary|verdict|conclusion|findings|claim.ledger|thesis|analysis|handoff|next.plan|reevaluation|profile.delta|audit|manifest)',re.I)
    for h,rr in sorted(tips.items()):
        if S.run(['git','cat-file','-e',h+'^{tree}'],cwd=ROOT,stdout=S.DEVNULL,stderr=S.DEVNULL).returncode:
            missing.append(dict(tip=h,refs=rr));continue
        for row in g('ls-tree','-r','-z',h,'docs','results').split(b'\0'):
            if not row:continue
            left,p=row.split(b'\t',1);p=p.decode(errors='surrogateescape');mode,kind,b=left.decode().split()
            if kind!='blob':continue
            if p.startswith('docs/research/') or p.startswith('docs/legacy/'):
                paths.setdefault((p,b),set()).add(h)
            basename=p.rsplit('/',1)[-1]
            if p.endswith(('.md','.json','.txt')) and names.search(basename) and ('/stages/' not in p):
                blobs[b].append((h,p))
    counts=C.Counter();total=0
    for b,loc in sorted(blobs.items()):
        data=g('cat-file','blob',b);total+=len(data)
        p=loc[0][1]
        row=dict(blob=b,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),locations=[dict(tip=h,path=p) for h,p in loc])
        if p.endswith('.json'):
            try:
                x=json.loads(data);row['summary_keys']=list(x)[:40] if isinstance(x,dict) else ['array',len(x)] if isinstance(x,list) else [type(x).__name__]
            except (ValueError,UnicodeError):row['summary_keys']=['unparsed']
            counts['json']+=1
        else:
            s=data.decode(errors='replace');lines=s.splitlines()
            row['title']=next((l for l in lines if l.startswith('# ')),lines[0] if lines else '')[:250]
            row['headings']=[l[:160] for l in lines if re.match(r'^#{1,3} ',l)][:24]
            rx=re.compile(r'결론|판정|미입증|실패|verdict|conclusion|not established|not admitted|unresolved|negative|demot|kill|pass|block',re.I)
            hit=[l.strip()[:300] for l in lines if rx.search(l)]
            row['evidence_excerpts']=(hit[:2]+hit[-3:]) if len(hit)>5 else hit
            counts['text']+=1
        result[b]=row
    out=dict(refs=refs,advertised_remote_heads=remote,tip_count=len(tips),missing=missing,unique_research_path_blobs=len(paths),research_paths=[dict(path=p,blob=b,tips=sorted(hh)) for (p,b),hh in sorted(paths.items())],summary_blob_count=len(blobs),summary_bytes_read=total,summary_counts=dict(counts),records=list(result.values()),scope='All named local/remote/tag tips available at inventory time and advertised remote heads; actual content read of selected report-like text/JSON files at those tips. Not exhaustive intermediate commit content, untracked files, archives or deleted refs. Later metacog13 explicitly separate.')
    (D/'HISTORY_INVENTORY.json').write_text(json.dumps(out,ensure_ascii=True,sort_keys=True,separators=(',',':'))+'\n')
    print({k:out[k] for k in ('tip_count','missing','unique_research_path_blobs','summary_blob_count','summary_bytes_read','summary_counts')})
    print('refs',len(refs),'remote',len(remote))
if __name__=='__main__':main()
