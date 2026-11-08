from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from PIL import Image,ImageDraw
import json,hashlib,re,shutil
R=Path(__file__).resolve().parent
D=R.parents[1]/'PPT交付'
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
orig=D/'TileLang_昇腾模调度_视觉精修版.pptx'
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert hashfile(orig)==hashfile(R/'source-user.pptx'),'Original changed'
with ZipFile(R/'candidate.pptx') as z:files={n:z.read(n) for n in z.namelist()}
report=[]
speech=(R/'speech_draft.md').read_text(encoding='utf8')
notes=dict((int(a),b) for a,b in re.findall(r'^## P(\d{2})[^\n]*\n([\s\S]*?)(?=^## |\Z)',speech,re.M))
assert len(notes)==24
for n in range(1,25):
 p=f'ppt/slides/slide{n}.xml';root=E.fromstring(files[p]);m=json.loads((R/f'manifest-{n:02}.json').read_text(encoding="utf8"))
 assert root.get('show')!='0'
 ids={a.get('name'):a.get('id') for a in root.findall('.//p:cNvPr',ns)}
 targets={a.get('id') for a in root.findall('.//p:cNvPr',ns)}
 for a in root.findall('.//p:spTgt',ns):assert a.get('spid') in targets
 count=len(root.findall('.//p:cTn[@nodeType="clickEffect"]',ns));assert count==m['clicks']
 assert len(re.findall(r'（点击\s*\d+',notes[n]))==count
 text=''.join(root.xpath('.//a:t/text()',namespaces=ns))
 for bad in ['工厂','货位','矩阵机','向量机','98.77','4.23','core.py'] :assert bad not in text,(n,bad)
 if n==23:assert '23.35%' in text and '90.77%' in text
 note_xml=E.fromstring(files[f'ppt/notesSlides/notesSlide{n}.xml']);assert len(''.join(note_xml.xpath('.//a:t/text()',namespaces=ns)))>20
 report.append({'slide':n,'clicks':count,'motion':sum(a['effect']=='motion' for a in m['animations'])})
 # Static end-state preview: apply exactly the motion deltas, leave original final PPT untouched.
 for a in m['animations']:
  if a['effect']!='motion':continue
  for sp in root.findall('.//p:sp',ns):
   named=sp.find('p:nvSpPr/p:cNvPr',ns)
   if named is not None and named.get('name')==a['name']:
    off=sp.find('p:spPr/a:xfrm/a:off',ns);off.set('x',str(int(off.get('x'))+round(a['dx']*9525)));off.set('y',str(int(off.get('y'))+round(a['dy']*9525)))
 files[p]=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
with ZipFile(R/'endstate.pptx','w',ZIP_DEFLATED) as z:
 for n,b in files.items():z.writestr(n,b)
summary={'originalUnchanged':True,'originalSha256':hashfile(orig),'finalSha256':hashfile(R/'candidate.pptx'),'slides':24,'clickGroups':sum(r['clicks'] for r in report),'motionEffects':sum(r['motion'] for r in report),'notes':24,'allReferencesValid':True,'nativeSlideshowVerified':False,'perSlide':report}
(R/'content-animation-check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
shutil.copyfile(R/'speech_draft.md',D/'三章重构版_演讲稿.md')
shutil.copyfile(R/'章节过渡技巧.md',D/'章节过渡技巧.md')
print(json.dumps({k:v for k,v in summary.items() if k!='perSlide'},ensure_ascii=False))

