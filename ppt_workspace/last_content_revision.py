from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
import xml.etree.ElementTree as E

root=Path('D:/Dev/tilelang/ppt_workspace')
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
for key,value in ns.items(): E.register_namespace(key,value)
old='GPU 也需要静态流水；warp scheduler 仍保留动态补位'
lines=['Hopper 也按任务分工、提前排流水','寄存器压力限制驻留工人；warp scheduler 仍动态补位']
with ZipFile(root/'validated-candidate.pptx') as z: files={n:z.read(n) for n in z.namelist()}
slide=E.fromstring(files['ppt/slides/slide6.xml'])
matches=[s for s in slide.findall('.//p:sp',ns) if ''.join(s.itertext()).find(old)>=0]
assert len(matches)==1
s=matches[0]
paragraph=s.find('.//a:p',ns)
run=paragraph.find('a:r',ns)
assert run is not None and run.find('a:t',ns).text==old
run.find('a:t',ns).text=lines[0]
paragraph.insert(list(paragraph).index(run)+1,E.Element('{'+ns['a']+'}br'))
second=deepcopy(run);second.find('a:t',ns).text=lines[1]
paragraph.insert(list(paragraph).index(run)+2,second)
x=s.find('./p:spPr/a:xfrm',ns)
x.find('a:off',ns).set('y',str(533*9525))
x.find('a:ext',ns).set('cy',str(78*9525))
files['ppt/slides/slide6.xml']=E.tostring(slide,encoding='utf-8',xml_declaration=True)
oldnote='GPU 也会用软件流水和任务分工，提前安排搬运与计算的重叠。静态安排与动态调度可以配合使用，工头补位这个机制仍然在。'
newnote='以 Hopper 上的大块矩阵计算为例，每个工人需要的工作台，也就是寄存器，可能很大，这会限制车间里能同时驻留多少工人。程序可以让一组工人专门搬运，另一组专门计算，提前排好流水，配合异步硬件，降低部分工人的寄存器需求。Hopper 不是静态流水的起点；这里要强调的是，按计划分工与工头动态补位可以同时存在。'
n='ppt/notesSlides/notesSlide6.xml'
txt=files[n].decode('utf-8');assert oldnote in txt
txt=txt.replace(oldnote,newnote)
txt=txt.replace('GPU机制：','Hopper参考：https://docs.nvidia.com/cuda/hopper-tuning-guide/；https://docs.nvidia.com/cutlass/4.5.2/media/docs/cpp/efficient_gemm.html。GPU机制：')
files[n]=txt.encode('utf-8')
with ZipFile(root/'validated-candidate-v3.pptx','w',ZIP_DEFLATED) as z:
    for name,b in files.items():z.writestr(name,b)
print('Updated only slide6 text box and speaker notes; retained all animation targets and chart data.')
