"""Make existing transition sentences the final manual build on each slide.

Only p:timing is changed. Text, shape geometry, formatting, other builds,
notes, media, and every other ZIP member are retained without rewriting.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

from lxml import etree as E

ROOT = Path(__file__).resolve().parent
TARGET = ROOT.parents[1] / 'PPT交付' / 'TileLang_昇腾算子.pptx'
BACKUP = ROOT / 'before-transition-animation.pptx'
CANDIDATE = ROOT / 'candidate.pptx'
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
Q = lambda name: '{' + NS['p'] + '}' + name
TIMING = re.compile(rb'<p:timing(?:\s[^>]*)?>.*?</p:timing>', re.S)
EFFECTS = 'p:cTn/p:childTnLst/p:par/p:cTn/p:childTnLst'
SENTENCES = {
    2: ('4', '为什么算力没有被充分利用？'),
    4: ('46', '怎样减少硬件空转？'),
    5: ('21', '如何实现？先看硬件的执行模型。'),
    7: ('31', '各个warp如何协同工作？'),
    8: ('26', '昇腾的执行模型怎样组织任务？'),
    11: ('18', 'TileLang编译后端自动排流水，支持昇腾算子高性能运行。'),
    12: ('5', '怎样把计算流程变成高效流水？'),
    13: ('15', 'TileLang拿到计算流程后，如何建模和分析？'),
    14: ('35', '长循环中，减小启动间隔II能直接缩短总时间。'),
    16: ('14', '有了对任务的控制，哪些条件限制了这些变量？'),
    21: ('21', '搜索可行的最小启动间隔，并取出对应调度变量。'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(node):
    return E.tostring(node, method='c14n')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not BACKUP.exists():
        BACKUP.write_bytes(TARGET.read_bytes())
    assert sha(TARGET) == sha(BACKUP), 'Source changed after backup; do not overwrite.'
    with ZipFile(BACKUP) as z:
        assert z.testzip() is None
        infos = z.infolist()
        original = {i.filename: z.read(i.filename) for i in infos}
    files = dict(original)
    donor_root = E.fromstring(files['ppt/slides/slide5.xml'])
    donor_timing = donor_root.find('p:timing', NS)
    donor_main = donor_timing.find('.//p:cTn[@nodeType="mainSeq"]/p:childTnLst', NS)
    donor_group = deepcopy(donor_main[1])
    donor_fade = deepcopy(donor_group.find(EFFECTS, NS)[0])
    assert donor_fade.find('p:cTn', NS).get('presetClass') == 'entr'
    audit = []

    for page, (sid, text) in SENTENCES.items():
        name = f'ppt/slides/slide{page}.xml'
        source = files[name]
        root = E.fromstring(source)
        shape = next(s for s in root.findall('p:cSld/p:spTree/p:sp', NS)
                     if s.find('p:nvSpPr/p:cNvPr', NS).get('id') == sid)
        assert ''.join(shape.xpath('.//a:t/text()', namespaces=NS)) == text
        timing = root.find('p:timing', NS)
        old_clicks = len(root.findall('.//p:cTn[@nodeType="clickEffect"]', NS))
        assert sid not in {n.get('spid') for n in root.findall('.//p:spTgt', NS)}
        if timing is None:
            timing = deepcopy(donor_timing)
            timing.find('.//p:cTn[@nodeType="mainSeq"]/p:childTnLst', NS).clear()
            timing.find('p:bldLst', NS).clear()
        main_seq = timing.find('.//p:cTn[@nodeType="mainSeq"]/p:childTnLst', NS)
        old_groups = [canonical(g) for g in main_seq]
        build_list = timing.find('p:bldLst', NS)
        old_builds = [canonical(b) for b in build_list]

        group = deepcopy(donor_group)
        group.find('p:cTn/p:stCondLst/p:cond', NS).set('delay', 'indefinite')
        effects = group.find(EFFECTS, NS)
        effects.clear()
        effect = deepcopy(donor_fade)
        top = effect.find('p:cTn', NS)
        top.set('nodeType', 'clickEffect')
        top.find('p:stCondLst/p:cond', NS).set('delay', '0')
        for n in effect.findall('.//p:cTn', NS):
            if n.get('dur') is not None:
                n.set('dur', '300')
        for n in effect.findall('.//p:spTgt', NS):
            n.set('spid', sid)
        effects.append(effect)
        last_id = max(int(n.get('id')) for n in timing.findall('.//p:cTn', NS))
        for i, n in enumerate(group.findall('.//p:cTn', NS), last_id + 1):
            n.set('id', str(i))
        main_seq.append(group)
        E.SubElement(build_list, Q('bldP'), spid=sid, grpId='0', animBg='1')

        assert [canonical(g) for g in list(main_seq)[:-1]] == old_groups
        assert [canonical(b) for b in list(build_list)[:-1]] == old_builds
        assert len(timing.findall('.//p:cTn[@nodeType="clickEffect"]', NS)) == old_clicks + 1
        assert {n.get('spid') for n in main_seq[-1].findall('.//p:spTgt', NS)} == {sid}
        replacement = E.tostring(timing, encoding='utf-8', with_tail=False)
        if TIMING.search(source):
            output, count = TIMING.subn(lambda _: replacement, source)
            assert count == 1
        else:
            assert source.endswith(b'</p:sld>') and root.find('p:extLst', NS) is None
            output = source[:-8] + replacement + b'</p:sld>'
        assert TIMING.sub(b'', output) == TIMING.sub(b'', source)
        files[name] = output
        audit.append({'page': page, 'shapeId': sid, 'text': text,
                      'oldClicks': old_clicks, 'newClicks': old_clicks + 1,
                      'effect': 'fade', 'durationMs': 300, 'trigger': 'last manual click'})

    with ZipFile(CANDIDATE, 'w', ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, files[info.filename])
    with ZipFile(CANDIDATE) as z:
        assert z.testzip() is None
        for name in z.namelist():
            data = z.read(name)
            assert TIMING.sub(b'', data) == TIMING.sub(b'', original[name])
            if name.endswith('.xml'):
                E.fromstring(data)
        for page in range(1, 25):
            name = f'ppt/slides/slide{page}.xml'
            root = E.fromstring(z.read(name))
            ids = {n.get('id') for n in root.findall('.//p:cNvPr', NS)}
            assert all(n.get('spid') in ids for n in root.findall('.//p:spTgt', NS))
            tids = [n.get('id') for n in root.findall('.//p:cTn', NS)]
            assert len(tids) == len(set(tids))
            if page not in SENTENCES:
                assert z.read(name) == original[name]
    report = {'sourceSha256': sha(BACKUP), 'candidateSha256': sha(CANDIDATE),
              'changedPages': list(SENTENCES), 'animations': audit,
              'allNonAnimationBytesPreserved': True, 'allExistingGroupsPreserved': True,
              'slideshowRun': False}
    (ROOT / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
