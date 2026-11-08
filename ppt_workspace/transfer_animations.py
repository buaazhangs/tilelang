from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
import json
from lxml import etree as E
root=Path('D:/Dev/tilelang/ppt_workspace')
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
with ZipFile(root/'before_visual_revision.pptx') as z: old={n:z.read(n) for n in z.namelist()}
with ZipFile(root/'visual-static-embedded-v4.pptx') as z: files={n:z.read(n) for n in z.namelist()}
report=[]
for i in range(1,24):
 name=f'ppt/slides/slide{i}.xml'
 prev=E.fromstring(old[name]);cur=E.fromstring(files[name])
 oldnames={x.get('id'):x.get('name') for x in prev.findall('.//p:cNvPr',ns)}
 newids={x.get('name'):x.get('id') for x in cur.findall('.//p:cNvPr',ns)}
 count=0
 for local in ['transition','timing']:
  node=prev.find('p:'+local,ns)
  if node is None: continue
  node=deepcopy(node)
  for x in node.iter():
   if x.get('spid') is not None:
    nm=oldnames[x.get('spid')]
    assert nm in newids,(i,nm)
    x.set('spid',newids[nm]);count+=1
  existing=cur.find('p:'+local,ns)
  if existing is not None:cur.remove(existing)
  ext=cur.find('p:extLst',ns)
  if ext is None:cur.append(node)
  else:cur.insert(list(cur).index(ext),node)
 if i>21:cur.set('show','0')
 files[name]=E.tostring(cur,xml_declaration=True,encoding='UTF-8',standalone=True)
 clicks=len(cur.findall('.//p:cTn[@nodeType="clickEffect"]',ns))
 manifest=json.loads((root/f'manifest-{i:02}.json').read_text(encoding='utf-8'))
 assert clicks==manifest['clicks'],(i,clicks,manifest['clicks'])
 report.append({'slide':i,'clicks':clicks,'retargetedReferences':count})
with ZipFile(root/'visual-animated-candidate.pptx','w',ZIP_DEFLATED) as z:
 for name,data in files.items():z.writestr(name,data)
(root/'visual-animation-audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('23 slides; preserved',sum(r['clicks'] for r in report),'click groups. Every animation reference mapped by content name.')
