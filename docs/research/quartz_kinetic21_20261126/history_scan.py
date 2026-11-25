"""Exhaustive reachable pre-COG commit and result-document inventory (no raw replay claim)."""
import subprocess as sp, pathlib, hashlib,json,collections,re
D=pathlib.Path(__file__).resolve().parent;ROOT=D.parents[2]
JUL='738ffe8ef165cf0d4e2c402258c85a63a0b30ffe'
NOV='6d279096c8b125c8af8d222784fac31a2d36b187'
def git(*a):return sp.check_output(['git','-C',str(ROOT),*a])
def text(*a):return git(*a).decode(errors='replace')
# Use current pre-existing refs; future K21 commits cannot enter date range.
refs=text('for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/remotes','refs/tags').splitlines()
heads={r.split()[0]:r.split()[1] for r in refs if 'kinetic21' not in r}
all_pre=set(text('rev-list','--all','--before=2026-11-20 00:00:00 +0900').splitlines())
j=set(text('rev-list',JUL).splitlines());n=set(text('rev-list',NOV).splitlines())
commits=[]
for h in sorted(all_pre|j|n):
    meta=text('show','-s','--format=%H%x09%P%x09%aI%x09%s',h).strip().split('\t',3)
    changed=text('diff-tree','--root','--no-commit-id','-m','-r','--name-only',h).splitlines()
    commits.append(dict(sha=meta[0],parents=meta[1].split(),date=meta[2],subject=meta[3],pre_first_meta=h in j,pre_modern_cog=h in n,changed=sorted(set(changed))))
# Every distinct historical blob of non-legacy research/report/analysis/experiment docs and code result manifests across these commits.
# Diff-path driven, include all root/canonical documents at both boundaries; avoid reading large unrelated binaries.
def wanted(p):
    if p.startswith('docs/legacy/'):return False
    lo=p.lower()
    return (lo.endswith(('.md','.tex','.rst')) and (p.startswith('docs/') or '/' not in p)) or (lo.endswith(('.json','.csv','.jsonl','.txt')) and any(k in lo for k in ('result','summary','manifest','evidence','verdict','audit','report')))
blobs={}
for row in commits:
    h=row['sha'];paths=[p for p in row['changed'] if wanted(p)]
    if not paths:continue
    # actual tree entries, not filename guesses
    for ln in text('ls-tree','-r',h,'--',*paths).splitlines():
        head,p=ln.split('\t',1);mode,typ,oid=head.split()
        if typ!='blob' or not wanted(p):continue
        key=(p,oid);blobs.setdefault(key,dict(path=p,blob=oid,example_commit=h,commits=[]))['commits'].append(h)
# read all distinct text blobs; preserve identity, headings, size and evidence-bearing lines, no raw outputs duplicated
for row in blobs.values():
    b=git('cat-file','blob',row['blob']);s=b.decode(errors='replace');lines=s.splitlines()
    row.update(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),line_count=len(lines),headings=[x[:300] for x in lines if re.match(r'^#{1,6}\s',x)][:80])
    if row['path'].endswith(('.md','.tex','.rst')):
        pat=re.compile(r'(?i)\b(pass(ed)?|fail(ed)?|kill(ed)?|falsif|verdict|deprecat|unresolved|regret|curvature|fisher|entropy|gibbs|covariance|bias|variance|duplicate|cost|support|reject|correct|saddlepoint|confidence|certificate|one.loop|phase|ablation|negative|tested)\b|검증|실패|판정|결론|결과|오류')
        hit=[(i+1,x[:500]) for i,x in enumerate(lines) if pat.search(x)]
        row['evidence_line_count']=len(hit);row['evidence_extract']=hit[:12]+(hit[-8:] if len(hit)>12 else [])
    else:
        try:
            q=json.loads(s);row['json_keys']=list(q)[:80] if isinstance(q,dict) else ['array',len(q)]
        except Exception:pass
out=dict(cutoffs=dict(july=JUL,november=NOV),ref_heads=heads,counts=dict(july=len(j),november=len(n),all_pre=len(all_pre),union=len(commits),side_only=len(all_pre-n),distinct_document_blobs=len(blobs),distinct_paths=len({x[0] for x in blobs})),commits=commits,documents=list(blobs.values()))
(D/'HISTORY_INDEX.json').write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(out['counts'])
print('historical side branches:')
for row in sorted((r for r in commits if not r['pre_modern_cog']),key=lambda r:r['date']):print(row['sha'][:8],row['date'][:10],row['subject'])
print('distinct results paths')
for p in sorted({x[0] for x in blobs}):
    if re.search(r'(?i)research|idea_foundry|phase15|audit_stage7|claim_ledger|quartz_theory|bq_plus|report\.md|audit_external',p) and p.endswith('.md'):print(p)
