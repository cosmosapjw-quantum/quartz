"""Preserve Matplotlib bytes, remove SVG line-end spaces, and verify XML tokens."""
from pathlib import Path
import gzip,hashlib,json,subprocess,xml.etree.ElementTree as ET,datetime
ROOT=Path(__file__).resolve().parents[4]
DOC=ROOT/'docs/research/quartz_general_incumbent_price_20261123'
EV=DOC/'evidence';SVG=DOC/'figures/C25_TEST_DECISIONS.svg'
def sha(x): return hashlib.sha256(x).hexdigest()
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert head=='d45173685f955820bd848bf4ddb2044382ca7e67','no C09 commit on failed preflight'
cmd=['git','diff','--cached','--check','--','.',':(exclude,glob)**/*.log',':(exclude,glob)**/*.stdout',':(exclude,glob)**/*.stderr']
p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
assert p.returncode==2
log=EV/'C09_SVG_PREFLIGHT_FAILED.log';assert not log.exists();log.write_bytes(p.stdout)
old=SVG.read_bytes();original=json.loads((EV/'C25_FIGURE.json').read_text())
assert sha(old)==original['artifacts']['figures/C25_TEST_DECISIONS.svg']['sha256']
raw=EV/'C25_TEST_DECISIONS_MATPLOTLIB.svg.gz';assert not raw.exists()
with raw.open('xb') as stream:
 with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as z:z.write(old)
assert gzip.decompress(raw.read_bytes())==old
new=b'\n'.join(line.rstrip(b' \t\r') for line in old.split(b'\n'))
a=list(ET.fromstring(old).iter());b=list(ET.fromstring(new).iter());assert len(a)==len(b)
for x,y in zip(a,b):
 assert x.tag==y.tag and x.attrib.keys()==y.attrib.keys()
 for k in x.attrib:assert x.attrib[k].split()==y.attrib[k].split(),(x.tag,k)
 assert (x.text or '').strip()==(y.text or '').strip()
 assert (x.tail or '').strip()==(y.tail or '').strip()
snapshot=EV/'C25_FIGURE_RENDER_ORIGINAL.json';assert not snapshot.exists()
snapshot.write_bytes((EV/'C25_FIGURE.json').read_bytes())
SVG.write_bytes(new)
original['artifacts']['figures/C25_TEST_DECISIONS.svg']={'sha256':sha(new),'bytes':len(new)}
original['packaging']={'normalization':'remove only trailing line whitespace; all parsed SVG attribute tokens/text and element order unchanged','script':'coding/normalize_c25_svg.py','script_sha256':sha(Path(__file__).read_bytes()),'original_svg_gz':str(raw.relative_to(DOC)),'original_svg_sha256':sha(old),'original_render_receipt':'evidence/C25_FIGURE_RENDER_ORIGINAL.json','scientific_inputs_changed':False}
(EV/'C25_FIGURE.json').write_text(json.dumps(original,indent=2)+'\n')
record={'status':'SVG_PACKAGING_CORRECTED_BEFORE_COMMIT','head_unchanged':head,'initial_preflight_exit':p.returncode,'preflight_log_sha256':sha(log.read_bytes()),'original_svg_sha256':sha(old),'packaged_svg_sha256':sha(new),'changed_lines':sum(x!=y for x,y in zip(old.splitlines(),new.splitlines())),'xml_elements_checked':len(a),'attribute_token_and_text_identity':True,'png_unchanged':sha((DOC/'figures/C25_TEST_DECISIONS.png').read_bytes())==original['artifacts']['figures/C25_TEST_DECISIONS.png']['sha256'],'scientific_inputs_changed':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
out=EV/'C09_SVG_PACKAGING.json';assert not out.exists();out.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
