"""Add native timing only; retain every user-owned non-timing XML byte."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from copy import deepcopy
import hashlib
import json
import re
from lxml import etree as E

ROOT = Path(__file__).resolve().parent
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
Q = lambda s: '{' + NS['p'] + '}' + s
TIMING = re.compile(rb'<p:timing(?:\s[^>]*)?>.*?</p:timing>', re.S)
GROUP_EFFECTS = 'p:cTn/p:childTnLst/p:par/p:cTn/p:childTnLst'


def automatic_intro(root, batches):
    timing = root.find('p:timing', NS)
    main = timing.find('.//p:cTn[@nodeType="mainSeq"]/p:childTnLst', NS)
    old_groups = [E.tostring(g) for g in main]
    old_clicks = len(timing.findall('.//p:cTn[@nodeType="clickEffect"]', NS))
    new_group = deepcopy(main[0])
    new_group.find('p:cTn/p:stCondLst/p:cond', NS).set('delay', '0')
    effects = new_group.find(GROUP_EFFECTS, NS)
    donor = deepcopy(effects[0])
    assert donor.find('p:cTn', NS).get('presetClass') == 'entr'
    effects.clear()
    existing_targets = {t.get('spid') for t in timing.findall('.//p:spTgt', NS)}
    actual_shapes = {s.get('id') for s in root.findall('p:cSld/p:spTree//p:cNvPr', NS)}
    build = timing.find('p:bldLst', NS)
    added = []
    for delay, ids in batches:
        for sid in map(str, ids):
            assert sid in actual_shapes and sid not in existing_targets
            effect = deepcopy(donor)
            top = effect.find('p:cTn', NS)
            top.set('nodeType', 'afterEffect' if not added else 'withEffect')
            top.find('p:stCondLst/p:cond', NS).set('delay', str(delay))
            for n in effect.findall('.//p:cTn', NS):
                if n.get('dur') is not None:
                    n.set('dur', '250')
            for n in effect.findall('.//p:spTgt', NS):
                n.set('spid', sid)
            effects.append(effect)
            E.SubElement(build, Q('bldP'), spid=sid, grpId='0', animBg='1')
            added.append(sid)
    maximum = max(int(n.get('id')) for n in timing.findall('.//p:cTn', NS))
    for i, n in enumerate(new_group.findall('.//p:cTn', NS), maximum + 1):
        n.set('id', str(i))
    main.insert(0, new_group)
    assert [E.tostring(g) for g in list(main)[1:]] == old_groups
    assert len(timing.findall('.//p:cTn[@nodeType="clickEffect"]', NS)) == old_clicks
    return {'addedTargets': added, 'manualClicks': old_clicks,
            'automaticDurationMs': max(d for d, _ in batches) + 250}


def fix_pv_endpoint(root, slide_width):
    shapes = {s.find('p:nvSpPr/p:cNvPr', NS).get('id'): s
              for s in root.findall('p:cSld/p:spTree/p:sp', NS)}
    softmax = shapes['10'].find('p:spPr/a:xfrm', NS)
    pv = shapes['23'].find('p:spPr/a:xfrm', NS)
    softmax_end = int(softmax.find('a:off', NS).get('x')) + int(softmax.find('a:ext', NS).get('cx'))
    pv_start = int(pv.find('a:off', NS).get('x'))
    # Leave a 20px gap after the producing softmax; do not move the source shape.
    endpoint = softmax_end + 20 * 9525
    motion = next(m for m in root.findall('.//p:animMotion', NS)
                  if m.find('.//p:spTgt', NS).get('spid') == '23')
    old_path = motion.get('path')
    path = f'M 0 0 L {(endpoint-pv_start)/slide_width:.12g} 0 E'
    motion.set('path', path)
    assert endpoint > softmax_end
    return {'target': '23', 'oldPath': old_path, 'newPath': path,
            'pvStartAtEndEMU': endpoint, 'softmaxEndEMU': softmax_end,
            'manualClicks': 4}


def main():
    with ZipFile(ROOT / 'normalized.pptx') as z:
        infos = z.infolist()
        original = {i.filename: z.read(i.filename) for i in infos}
    files = dict(original)
    pres = E.fromstring(files['ppt/presentation.xml'])
    width = int(pres.find('p:sldSz', NS).get('cx'))
    report = {}
    for page in (4, 5, 7):
        name = f'ppt/slides/slide{page}.xml'
        root = E.fromstring(files[name])
        if page == 4:
            report[page] = automatic_intro(root, [(0, range(6, 21)), (250, range(21, 35))])
        elif page == 7:
            report[page] = automatic_intro(root, [(0, [*range(5, 15), 34])])
        else:
            report[page] = fix_pv_endpoint(root, width)
        timing = root.find('p:timing', NS)
        replacement = E.tostring(timing, encoding='utf-8', with_tail=False)
        data, count = TIMING.subn(lambda _: replacement, files[name])
        assert count == 1
        assert TIMING.sub(b'', data) == TIMING.sub(b'', original[name])
        ids = [n.get('id') for n in timing.findall('.//p:cTn', NS)]
        assert len(ids) == len(set(ids))
        files[name] = data
    changed = [n for n in files if files[n] != original[n]]
    assert changed == ['ppt/slides/slide4.xml', 'ppt/slides/slide5.xml', 'ppt/slides/slide7.xml'] or set(changed) == {f'ppt/slides/slide{p}.xml' for p in (4,5,7)}
    # Every other payload, including masters, media, relationships, and notes, is identical.
    output = ROOT / 'candidate.pptx'
    with ZipFile(output, 'w', ZIP_DEFLATED) as z:
        for info in infos:
            z.writestr(info, files[info.filename])
    with ZipFile(output) as z:
        assert z.testzip() is None
        for n, data in files.items():
            assert z.read(n) == data
        for n in files:
            if n.endswith('.xml'):
                E.fromstring(files[n])
    report['changedParts'] = changed
    report['nonTimingXmlIdentical'] = True
    report['allOtherPayloadsIdentical'] = True
    report['sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    (ROOT / 'changes.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
