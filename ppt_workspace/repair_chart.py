from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as E
root=Path('D:/Dev/tilelang/ppt_workspace')
ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
# WPS rewrites the series title's cell reference as a literal string in c:f.
# Restore its real embedded worksheet cell, preserving cached values and all
# visual/animation XML. The numeric category/value references already match.
with ZipFile(root/'animated-draft.pptx') as z:
    files={n:z.read(n) for n in z.namelist()}
name='ppt/charts/chart1.xml'
text=files[name].decode('utf-8')
tree=E.fromstring(text)
f=tree.find('.//c:ser/c:tx/c:strRef/c:f',ns)
assert f is not None
old=f.text
assert old != "'Chart Data'!$B$1"
import re
text,n=re.subn(r'(<c:tx><c:strRef><c:f>).*?(</c:f>)',r"\g<1>'Chart Data'!$B$1\g<2>",text,count=1)
assert n==1
files[name]=text.encode('utf-8')
with ZipFile(root/'validated-candidate.pptx','w',ZIP_DEFLATED) as z:
    for n,b in files.items():z.writestr(n,b)
print('Restored chart title reference to embedded Chart Data!B1')
