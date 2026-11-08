from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
import hashlib, json, re

R = Path(__file__).resolve().parent.parent
D = R.parents[1] / 'PPT交付'
A = R / 'playback-audit'
ns = {'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
old = D / 'TileLang_昇腾模调度_视觉精修版.pptx'
new = D / 'TileLang_昇腾模调度_三章重构版.pptx'
assert sha(old) == '43b78aa61ddf016e349946b76e08f4c4c5fb1d596c684dbc0cfdbea1204b55e3'
assert sha(new) == sha(A / 'checked.pptx')
speech = (R / 'speech_draft.md').read_text(encoding='utf-8')
assert speech == (D / '三章重构版_演讲稿.md').read_text(encoding='utf-8')
notes = dict((int(n), t.strip()) for n,t in re.findall(r'^## P(\d{2})[^\n]*\n([\s\S]*?)(?=^## |\Z)', speech, re.M))
counts = [0,0,4,3,3,0,3,4,6,4,3,0,3,4,3,3,3,3,3,3,4,4,3,0]
observations = [
 '封面正常；口播后翻页。',
 '第一章切换页正常；无额外动画点击。',
 '输入搬运、QK与搬出、softmax与回传、PV与写回，四组依次出现。',
 '串行任务、25%忙碌比例、432→108换算依次出现。',
 'K₁真实移动到另一批计算期间；随后出现重排与多缓冲说明。',
 '第二章切换页正常；无额外动画点击。',
 '同质warp、专职warp、指令流概念依次出现。',
 '等待、两侧就绪任务选择、总结依次出现。',
 '六条Scalar指令逐次移入对应单元；文字、落点和讲稿一致。',
 '修复QK₁/PV₀途中遮挡；上下错开换位，结束后显示总结；已原生复放。',
 '长B、四个A、约束说明依次出现；讲稿两处动作措辞已修正。',
 '第三章切换页正常；无额外动画点击。',
 '用户流程、编译器工作、连接箭头三步对应讲稿。',
 '利用率、重叠批次、II、总时间公式与标尺四步对应。',
 '阶段k、位置r、buffer份数b依次出现。',
 '未知变量、已知时间、开始时间公式依次出现。',
 '生产消费图、同轮约束、跨轮约束依次出现。',
 '本轮写入/最后读取、下一次写入、容量限制依次出现；首步指图措辞已修正。',
 '占用区间、i在j前、j在i前三步对应。',
 '约束汇总与三色、未知量乘积、求解成本三步对应。',
 '固定II、Z3、可行分支、不可行分支四步对应。',
 '串行指令、重排指令、buffer写入、读取与同步四步对应。',
 '串行图、流水图、MFU三步对应；修复左侧两个归一标签断行并原生复放。',
 '结束页正常；讲稿要求停止点击并进入提问。',
]
per_slide = []
with ZipFile(new) as z:
    assert len([p for p in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml', p)]) == 24
    for n in range(1,25):
        root = E.fromstring(z.read(f'ppt/slides/slide{n}.xml'))
        click = len(root.findall('.//p:cTn[@nodeType="clickEffect"]', ns))
        motion = len(root.findall('.//p:animMotion', ns))
        assert click == counts[n-1]
        assert root.get('show') != '0'
        assert len(re.findall(r'（点击\s*\d+', notes[n])) == click
        note = E.fromstring(z.read(f'ppt/notesSlides/notesSlide{n}.xml'))
        actual = '\n'.join(''.join(p.xpath('.//a:t/text()', namespaces=ns)) for p in note.findall('.//a:p',ns)).strip()
        assert actual == notes[n], n
        ids = {p.get('id') for p in root.findall('.//p:cNvPr',ns)}
        assert all(p.get('spid') in ids for p in root.findall('.//p:spTgt',ns))
        per_slide.append({'slide':n,'clicks':click,'motion':motion,'nativeWpsPlaybackChecked':True,'observation':observations[n-1]})
assert sum(x['clicks'] for x in per_slide) == 66
assert sum(x['motion'] for x in per_slide) == 9

summary = {
 'originalUnchanged': True, 'originalSha256':sha(old), 'finalSha256':sha(new),
 'finalPath':str(new), 'slides':24, 'clickGroups':66, 'motionEffects':9,
 'notes':24, 'allNotesMatchSpeech':True, 'allReferencesValid':True,
 'nativeSlideshowVerified':True, 'nativeApplication':'WPS Presentation 12.1.0.28043',
 'nativePowerPointVerified':False, 'timedOralRehearsalPerformed':False,
 'replayedAfterRepair':[10,23], 'perSlide':per_slide,
}
(A / 'native-playback-check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(R / 'content-animation-check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
repair = json.loads((A / 'repair-report.json').read_text(encoding='utf-8'))
repair.update(nativePlaybackVerified=True, nativeApplication=summary['nativeApplication'], finalPath=str(new), finalSha256=sha(new))
(A / 'repair-report.json').write_text(json.dumps(repair,ensure_ascii=False,indent=2),encoding='utf-8')

report = '''# 三章重构版：实际放映检查

已在本机 WPS Presentation 12.1.0.28043 中，从封面到结束页实际逐次放映24页、66组页内点击，并对照每页演讲稿。检查了初始状态、每次出现内容、移动过程与落点。修复后重新打开交付文件，复放第10页和第23页。

## 修复结果

- 第09页：六次Scalar指令移动均正常，依次进入搬运、Cube、Vector的相应位置；讲稿逐次对应，无须修改。
- 第10页：原来QK₁与PV₀沿同一水平线交换，途中相互遮挡。改为上下错开的路径，1.2秒完成换位，随后显示结论；仍为原来的第4次点击，不增加点击。复放时检查了中途分离状态和最终QK₀→QK₁→PV₀顺序。
- 第11页：四个A是同次点击一起出现，将“逐个显示”改为“显示后逐个指读”；把不存在的“依赖箭头”改为实际画面中的“任务与下方约束说明”。
- 第18页：第一次点击只显示本轮写入与最后读取，讲稿指图动作已据此修正；下一次写入仍在第二次点击出现。
- 第23页：左侧归一₀、归一₁的下标意外换行，已通过减小内边距、保持单行修复，字号不变；WPS复放确认。

三处讲稿动作修正已同步到独立演讲稿、24页PPT备注及在线版相应备注。全部点击数和页序保持不变。

## 逐页记录

| 页码 | 页内点击数 | 实际检查结果 |
|---|---:|---|
'''
report += '\n'.join(f"| {n:02} | {counts[n-1]} | {observations[n-1]} |" for n in range(1,25))
report += f'''

页内点击数不包含翻页动作。封面、三张章节页和结束页没有额外动画点击。

## 文件与检查范围

修复后的交付文件仍为 `TileLang_昇腾模调度_三章重构版.pptx`，配套讲稿为 `三章重构版_演讲稿.md`。原版 `TileLang_昇腾模调度_视觉精修版.pptx` 未修改，SHA256与检查前一致。

- 原版SHA256：`{sha(old)}`
- 修复版SHA256：`{sha(new)}`

本次实际使用WPS放映，未声称在Microsoft PowerPoint中实测，也未进行35–40分钟完整口播计时。文件结构、动画引用及24页备注与讲稿全文一致性均已另行核对。

检查前的新版备份及机器可读记录保存在 `../ppt_workspace/revision2/playback-audit/`。
'''
(D / '三章重构版_放映检查.md').write_text(report,encoding='utf-8')
check = D / '三章重构版_修改核对.md'
t = check.read_text(encoding='utf-8')
t = t.replace('已检查移动终点的静态画面，尚未在原生软件中完成全套动态放映；静态渲染与OOXML校验不等于实机播放验证。', '已在WPS Presentation 12.1.0.28043中实际逐次放映全部24页、66组点击，并在修复后复放第10页与第23页。Scalar指令移动通过；QK/PV换序改为上下错开，修复途中遮挡；同步修正第11、18页讲稿动作及第23页标签断行。详见《三章重构版_放映检查.md》。未进行Microsoft PowerPoint实测或完整口播计时。')
t = t.replace('b8f7040f14ab30dc0f6da289aafa6ad13c03eddd187b708a942ec512bd247685',sha(new))
check.write_text(t,encoding='utf-8')
with (R.parents[1] / 'GPT vs claude.md').open('a',encoding='utf-8') as f:
    f.write(f'\n\n## GPT · 三章重构版原生放映检查完成\n已在WPS Presentation 12.1.0.28043逐次放映24页、66组点击。P09六次Scalar移动正常；P10实际发现同水平线交换会遮挡，已改上下错开1.2秒路径，结论延后至换位结束，原生复放通过。P11两处/P18一处讲稿动作已同步到PPT备注及演讲稿；P23两个归一下标断行已修复并复放。交付文件名不变，SHA256 `{sha(new)}`。原视觉精修版SHA仍为 `{sha(old)}`。详情见PPT交付/三章重构版_放映检查.md；此轮无须等待回复。\n')
print(json.dumps({k:v for k,v in summary.items() if k!='perSlide'},ensure_ascii=False,indent=2))
