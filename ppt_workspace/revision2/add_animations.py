"""Inject click-controlled native animations into the revision 2 PPTX.

Uses the previously exported native OOXML timing tree as a template.  No COM,
no rewrite of shapes, media, notes or content.  Motion coordinates are pixels
on the source 1280 x 720 canvas.  Motion shapes remain visible before movement;
use a separate earlier fade entry if the shape should initially be hidden.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree as E

ROOT = Path(__file__).resolve().parent
NS = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}


def q(local):
    return "{" + NS["p"] + "}" + local


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "static.pptx")
    ap.add_argument("--output", type=Path, default=ROOT / "candidate.pptx")
    ap.add_argument("--manifest-dir", type=Path, default=ROOT)
    ap.add_argument("--template", type=Path, default=ROOT.parent / "before_visual_revision.pptx")
    ap.add_argument("--width", type=float, default=1280)
    ap.add_argument("--height", type=float, default=720)
    args = ap.parse_args()
    assert args.input.resolve() != args.output.resolve(), "Never overwrite static input"
    assert args.template.resolve() != args.output.resolve(), "Never overwrite donor deck"
    input_sha, template_sha = sha(args.input), sha(args.template)

    with ZipFile(args.template) as z:
        donor = E.fromstring(z.read("ppt/slides/slide4.xml"))
    timing_template = donor.find("p:timing", NS)
    assert timing_template is not None
    main_template = timing_template.find('.//p:cTn[@nodeType="mainSeq"]/p:childTnLst', NS)
    group_template = main_template[0]
    effects_template = group_template.find("p:cTn/p:childTnLst/p:par/p:cTn/p:childTnLst", NS)
    motion_template = effects_template[0]
    fade_template = effects_template[1]

    with ZipFile(args.input) as z:
        files = {n: z.read(n) for n in z.namelist()}
    slide_names = sorted(
        (n for n in files if n.startswith("ppt/slides/slide") and n.endswith(".xml")
         and "/_rels/" not in n),
        key=lambda n: int(Path(n).stem[5:]),
    )
    report = []
    for file_name in slide_names:
        i = int(Path(file_name).stem[5:])
        cur = E.fromstring(files[file_name])
        manifest = json.loads((args.manifest_dir / f"manifest-{i:02}.json").read_text(encoding="utf-8-sig"))
        groups = defaultdict(list)
        shape_ids = {x.get("name"): x.get("id") for x in cur.findall(".//p:cNvPr", NS)}
        for effect in manifest.get("animations", []):
            assert effect["name"] in shape_ids, (i, "missing shape", effect["name"])
            assert effect["effect"] in ("fade", "motion", "exit"), (i, effect)
            assert int(effect["click"]) > 0, (i, effect)
            groups[int(effect["click"])].append(effect)
        assert sorted(groups) == list(range(1, len(groups) + 1)), (i, "noncontiguous clicks", list(groups))
        if "clicks" in manifest:
            assert manifest["clicks"] == len(groups), (i, manifest["clicks"], len(groups))

        for local in ("transition", "timing"):
            old = cur.find("p:" + local, NS)
            if old is not None:
                cur.remove(old)
        cur.attrib.pop("show", None)
        transition = E.Element(q("transition"), advClick="1")
        ext = cur.find("p:extLst", NS)
        cur.insert(list(cur).index(ext) if ext is not None else len(cur), transition)

        timing = deepcopy(timing_template)
        main = timing.find('.//p:cTn[@nodeType="mainSeq"]/p:childTnLst', NS)
        main.clear()
        bld = timing.find("p:bldLst", NS)
        bld.clear()
        build_ids = set()
        effect_report = []
        for click in sorted(groups):
            group = deepcopy(group_template)
            effect_list = group.find("p:cTn/p:childTnLst/p:par/p:cTn/p:childTnLst", NS)
            effect_list.clear()
            for index, effect in enumerate(groups[click]):
                is_motion = effect["effect"] == "motion"
                node = deepcopy(motion_template if is_motion else fade_template)
                top = node.find("p:cTn", NS)
                top.set("nodeType", "clickEffect" if index == 0 else "withEffect")
                top.set("grpId", "0")
                target_id = shape_ids[effect["name"]]
                for target in node.findall(".//p:spTgt", NS):
                    target.set("spid", target_id)
                seconds = float(effect.get("duration", 0.65 if is_motion else 0.28))
                assert seconds > 0
                delay_seconds = float(effect.get("delay", 0))
                assert delay_seconds >= 0
                # Delay is relative to this click group, in seconds.  Apply it
                # at the effect root so both visibility and fade wait together.
                start = top.find("p:stCondLst/p:cond", NS)
                assert start is not None
                start.set("delay", str(round(delay_seconds * 1000)))
                for tn in node.findall(".//p:cTn", NS):
                    if tn.get("dur") is not None:
                        tn.set("dur", str(round(seconds * 1000)))
                if is_motion:
                    # Motion is not an entrance. Keep the instruction visible
                    # in Scalar before the click, then hold its moved position.
                    top.set("presetClass", "path")
                    behavior_list = top.find("p:childTnLst", NS)
                    for set_node in list(behavior_list.findall("p:set", NS)):
                        behavior_list.remove(set_node)
                    motion = node.find(".//p:animMotion", NS)
                    points = effect.get("pathPoints")
                    if points is None:
                        points = [[0, 0], [effect.get("dx", 0), effect.get("dy", 0)]]
                    assert len(points) >= 2 and len(points[0]) == 2
                    assert float(points[0][0]) == 0 and float(points[0][1]) == 0
                    segments = []
                    for point_index, point in enumerate(points):
                        assert len(point) == 2
                        x = float(point[0]) / args.width
                        y = float(point[1]) / args.height
                        segments.append(f"{'M' if point_index == 0 else 'L'} {x:.10g} {y:.10g}")
                    motion.set("path", " ".join(segments) + " E")
                    motion.set("origin", "layout")
                    motion.set("pathEditMode", "relative")
                    motion.find("p:cBhvr/p:cTn", NS).set("fill", "hold")
                elif effect["effect"] == "exit":
                    top.set("presetClass", "exit")
                    node.find(".//p:animEffect", NS).set("transition", "out")
                    # Use native fade-out then hold hidden after the duration.
                    visibility = node.find(".//p:set", NS)
                    visibility.find("p:to/p:strVal", NS).set("val", "hidden")
                    vis_tn = visibility.find("p:cBhvr/p:cTn", NS)
                    vis_tn.set("dur", "1")
                    vis_tn.find("p:stCondLst/p:cond", NS).set("delay", str(round(seconds * 1000)))
                effect_list.append(node)
                # Paragraph builds are useful for text/card backgrounds, but
                # must not target a picture or a group shape.
                target_shape = next((s for s in cur.findall(".//p:sp", NS)
                    if s.find("p:nvSpPr/p:cNvPr", NS) is not None
                    and s.find("p:nvSpPr/p:cNvPr", NS).get("id") == target_id), None)
                if target_shape is not None and target_id not in build_ids:
                    E.SubElement(bld, q("bldP"), spid=target_id, grpId="0", animBg="1")
                    build_ids.add(target_id)
                effect_report.append({"name": effect["name"], "click": click,
                    "effect": effect["effect"], "targetId": target_id,
                    "durationSeconds": seconds, "delaySeconds": delay_seconds,
                    **({"pathPoints": effect["pathPoints"]} if "pathPoints" in effect else {})})
            main.append(group)
        if len(bld) == 0:
            timing.remove(bld)
        # All timing IDs are unique within this slide. Templates have no tn-id
        # references: the main sequence is controlled by slide onNext/onPrev.
        for num, node in enumerate(timing.findall(".//p:cTn", NS), 1):
            node.set("id", str(num))
        if groups:
            ext = cur.find("p:extLst", NS)
            cur.insert(list(cur).index(ext) if ext is not None else len(cur), timing)
        files[file_name] = E.tostring(cur, xml_declaration=True, encoding="UTF-8", standalone=True)

        known_ids = set(shape_ids.values())
        dangling = [n.get("spid") for n in cur.iter()
                    if n.get("spid") is not None and n.get("spid") not in known_ids]
        assert not dangling, (i, dangling)
        clicks = len(cur.findall('.//p:cTn[@nodeType="clickEffect"]', NS))
        assert clicks == len(groups), (i, clicks, len(groups))
        ids = [n.get("id") for n in cur.findall(".//p:cTn", NS)]
        assert len(ids) == len(set(ids)), (i, "duplicate timing ids")
        report.append({"slide": i, "clicks": clicks, "effects": len(effect_report),
            "motionEffects": sum(e["effect"] == "motion" for e in effect_report),
            "hidden": False, "details": effect_report})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "w", ZIP_DEFLATED) as z:
        for n, data in files.items():
            z.writestr(n, data)
    assert sha(args.input) == input_sha and sha(args.template) == template_sha
    audit = {"inputSha256": input_sha, "templateSha256": template_sha,
        "outputSha256": sha(args.output), "slides": len(report),
        "totalClickGroups": sum(r["clicks"] for r in report),
        "totalEffects": sum(r["effects"] for r in report),
        "totalMotionEffects": sum(r["motionEffects"] for r in report),
        "nativeSlideshowVerified": False, "perSlide": report}
    (args.output.parent / "animation-audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "perSlide"}, indent=2))


if __name__ == "__main__":
    main()
