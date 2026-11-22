"""Reassemble the exact large TEST trace from versioned byte parts."""
import hashlib,json,pathlib,tempfile,os
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 ev=pathlib.Path(__file__).resolve().parents[1]/'evidence'
 spec=json.loads((ev/'C23_TEST_TRACE.parts.json').read_text());out=ev/spec['original_name']
 if out.exists():
  assert digest(out)==spec['original_sha256'],'existing trace identity mismatch'
  print(str(out));return
 tmp=None
 try:
  with tempfile.NamedTemporaryFile(dir=ev,prefix='c23_trace_',suffix='.tmp',delete=False) as f:
   tmp=pathlib.Path(f.name);whole=hashlib.sha256()
   for part in spec['parts']:
    p=ev/part['name'];assert p.stat().st_size==part['bytes'] and digest(p)==part['sha256']
    with p.open('rb') as source:
     for chunk in iter(lambda:source.read(1024*1024),b''):
      f.write(chunk);whole.update(chunk)
  assert whole.hexdigest()==spec['original_sha256']
  os.link(tmp,out)
 finally:
  if tmp is not None:tmp.unlink(missing_ok=True)
 print(str(out))
if __name__=='__main__':main()
