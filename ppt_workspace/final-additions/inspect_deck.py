from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import hashlib, json, shutil

R=Path(__file__).resolve().parent
D=R.parents[1]/'PPT交付'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inspect(path):
 with ZipFile(path) as z:
  pres=E.fromstring(z.read('ppt/presentation.xml'))
  rel=E.fromstring(z.read('ppt/_rels/presentation.xml.rels'))
  rels={x.get('Id'):x.get('Target') for x in rel}
  result=[]
  for page,item in enumerate(pres.find('p:sldIdLst',NS),1):
   part='ppt/'+rels[item.get('{'+NS['r']+'}id')].lstrip('/')
   if part.startswith('ppt/ppt/'):part=part[4:]
   root=E.fromstring(z.read(part))
   animated={t.get('spid') for t in root.findall('.//p:spTgt',NS)}
   shapes=[]
   def walk(parent,group=None):
    for s in parent:
     kind=E.QName(s).localname
     if kind not in ['sp','pic','cxnSp','graphicFrame','grpSp']:continue
     nv=s.find('.//p:cNvPr',NS)
     sid=nv.get('id'); name=nv.get('name')
     shape={'id':sid,'name':name,'kind':kind,'parentGroup':group,'text':' | '.join(s.xpath('.//a:t/text()',namespaces=NS)), 'animated':sid in animated, 'groupAnimated':group in animated}
     xform=s.find('p:spPr/a:xfrm',NS)
     if xform is None:xform=s.find('p:grpSpPr/a:xfrm',NS)
     if xform is not None:shape['xfrm']={E.QName(c).localname:dict(c.attrib) for c in xform}
     cr=nv.xpath('.//*[local-name()="creationId"]')
     if cr:shape['creationId']=cr[0].get('id')
     shapes.append(shape)
     if kind=='grpSp':walk(s,sid)
   walk(root.find('p:cSld/p:spTree',NS))
   groups=[]
   main=root.find('.//p:cTn[@nodeType="mainSeq"]/p:childTnLst',NS)
   if main is not None:
    for i,g in enumerate(main,1):
     effects=[]
     for eff in g.findall('.//p:cTn',NS):
      if eff.get('nodeType') not in ['clickEffect','withEffect','afterEffect']:continue
      targets=list(dict.fromkeys(t.get('spid') for t in eff.findall('.//p:spTgt',NS)))
      effects.append({'tnid':eff.get('id'),'nodeType':eff.get('nodeType'),'class':eff.get('presetClass'),'targets':targets,'text':[x['text'] for t in targets for x in shapes if x['id']==t]})
     groups.append({'click':i,'effects':effects})
   result.append({'page':page,'part':part,'shapes':shapes,'groups':groups})
  return {'path':str(path),'sha256':sha(path),'slides':result}

target=D/'TileLang_昇腾算子.pptx'
backup=R/'before-additions.pptx'
if not backup.exists():shutil.copyfile(target,backup)
assert sha(target)==sha(backup),'source changed since backup'
current=inspect(R/'normalized.pptx')
previous=inspect(D/'TileLang_昇腾模调度_三章重构版.pptx')
(R/'current.json').write_text(json.dumps(current,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'previous.json').write_text(json.dumps(previous,ensure_ascii=False,indent=2),encoding='utf-8')
out=[]
for s in current['slides']:
 old=previous['slides'][s['page']-1] if s['page']<=len(previous['slides']) else {'shapes':[]}
 oldnames={x['name']:x for x in old['shapes']}
 new=[x for x in s['shapes'] if x['name'] not in oldnames]
 unanimated=[x for x in new if not x['animated'] and not x['groupAnimated']]
 out.append({'page':s['page'],'part':s['part'],'clickGroups':len(s['groups']),'newElements':new,'newUnanimated':unanimated})
(R/'added-shapes.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'sourceSha256':current['sha256'],'pages':len(current['slides']),'summary':[{'page':x['page'],'clicks':x['clickGroups'],'newUnanimated':[{'id':s['id'],'name':s['name'],'kind':s['kind'],'text':s['text'],'parentGroup':s['parentGroup'],'xfrm':s.get('xfrm')} for s in x['newUnanimated']]} for x in out if x['newUnanimated']]},ensure_ascii=False,indent=2))
