"""Repair only observed P10 motion/summary timing and P11/P18 notes.

This script is intentionally prepared but not run by the audit agent.  The
parent runs it after the ongoing original-deck playback has finished.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import hashlib
import json
from lxml import etree as E

ROOT = Path(__file__).resolve().parent
NS = {'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(root):
    return E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=ROOT / 'before-playback.pptx')
    parser.add_argument('--output', type=Path, default=ROOT / 'repaired.pptx')
    args = parser.parse_args()
    assert args.input.resolve() != args.output.resolve()
    before_sha = sha(args.input)
    with ZipFile(args.input) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    original = dict(files)
    pres = E.fromstring(files['ppt/presentation.xml'])
    size = pres.find('p:sldSz', NS)
    assert (int(size.get('cx')), int(size.get('cy'))) == (12192000, 6858000)
    width, height = 1280.0, 720.0

    root = E.fromstring(files['ppt/slides/slide10.xml'])
    shapes = {}
    for shape in root.findall('.//p:sp', NS):
        node = shape.find('p:nvSpPr/p:cNvPr', NS)
        shapes[node.get('id')] = ''.join(shape.xpath('.//a:t/text()', namespaces=NS))
    main_sequence = root.find('.//p:cTn[@nodeType="mainSeq"]/p:childTnLst', NS)
    assert len(main_sequence) == 4
    click4 = main_sequence[3]
    motions = click4.findall('.//p:animMotion', NS)
    assert len(motions) == 2
    path_report = []
    for motion in motions:
        target = motion.find('.//p:spTgt', NS).get('spid')
        label = shapes[target]
        assert label in ('PV₀', 'QK₁'), label
        dx, dy = (386, 80) if label == 'PV₀' else (-386, -80)
        points = [(0, 0), (0, dy), (dx, dy), (dx, 0)]
        path = ' '.join(f'{"M" if i == 0 else "L"} {x/width:.10g} {y/height:.10g}'
                        for i, (x, y) in enumerate(points)) + ' E'
        motion.set('path', path)
        motion.set('origin', 'layout')
        motion.set('pathEditMode', 'relative')
        motion.find('p:cBhvr/p:cTn', NS).set('dur', '1200')
        path_report.append({'label': label, 'pathPointsPx': points, 'path': path,
                            'durationMs': 1200})

    effect_roots = click4.findall('.//p:cTn[@nodeType="withEffect"]', NS)
    summary_roots = [node for node in effect_roots
                     if any(shapes.get(t.get('spid')) == '利用softmax₀执行期间推进QK₁'
                            for t in node.findall('.//p:spTgt', NS))]
    assert len(summary_roots) == 1
    summary_roots[0].find('p:stCondLst/p:cond', NS).set('delay', '1200')
    assert len(root.findall('.//p:cTn[@nodeType="clickEffect"]', NS)) == 4
    files['ppt/slides/slide10.xml'] = encode(root)

    note_changes = {
        11: [
            ('（点击 2；逐个显示 A₀ 到 A₃，沿色块指读。）',
             '（点击 2；显示 A₀ 到 A₃，沿色块逐个指读。）'),
            ('（点击 3；指不同长度与依赖箭头。）',
             '（点击 3；指不同长度的任务与下方约束说明。）'),
        ],
        18: [
            ('（点击 1；指同一 buffer 版本的两次写入。）',
             '（点击 1；指本轮写入与最后一次读取。）'),
        ],
    }
    for n, replacements in note_changes.items():
        part = f'ppt/notesSlides/notesSlide{n}.xml'
        note = E.fromstring(files[part])
        for old, new in replacements:
            matches = [t for t in note.findall('.//a:t', NS) if old in (t.text or '')]
            assert len(matches) == 1, (n, old, len(matches))
            matches[0].text = matches[0].text.replace(old, new)
        files[part] = encode(note)

    # Native WPS playback wrapped only the two narrow left-side normalization labels.
    part = 'ppt/slides/slide23.xml'
    slide23 = E.fromstring(files[part])
    repaired_labels = []
    for shape in slide23.findall('.//p:sp', NS):
        shape_id = shape.find('p:nvSpPr/p:cNvPr', NS).get('id')
        if shape_id in ('15', '19'):
            label = ''.join(shape.xpath('.//a:t/text()', namespaces=NS))
            assert label in ('归一₀', '归一₁'), label
            body = shape.find('p:txBody/a:bodyPr', NS)
            body.set('lIns', '19050')
            body.set('rIns', '19050')
            body.set('wrap', 'none')
            repaired_labels.append(label)
    assert len(repaired_labels) == 2
    files[part] = encode(slide23)

    changed_parts = [n for n in files if files[n] != original[n]]
    assert sorted(changed_parts) == sorted([
        'ppt/slides/slide10.xml', 'ppt/notesSlides/notesSlide11.xml',
        'ppt/notesSlides/notesSlide18.xml', 'ppt/slides/slide23.xml'])
    click_count = 0
    for n in range(1, 25):
        slide = E.fromstring(files[f'ppt/slides/slide{n}.xml'])
        click_count += len(slide.findall('.//p:cTn[@nodeType="clickEffect"]', NS))
    assert click_count == 66
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    assert sha(args.input) == before_sha
    report = {'inputSha256': before_sha, 'outputSha256': sha(args.output),
              'changedParts': changed_parts, 'totalClickGroups': click_count,
              'p10ClickGroups': 4, 'motions': path_report,
              'summaryDelayMs': 1200, 'speakerNoteReplacements': 3,
              'singleLineLabels': repaired_labels,
              'nativePlaybackVerified': False}
    (args.output.parent / 'repair-report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
