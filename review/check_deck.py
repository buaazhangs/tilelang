#!/usr/bin/env python
"""Claude 侧的 PPT 机械校验脚本（复核用，只读，不参与制作）。

适配 GPT 的产物格式：
    ppt_workspace/manifest-NN.json          点击归属
    ppt_workspace/renders/slide-N.layout.json  元素坐标 / 文字 / 字号

用法:
    python review/check_deck.py            # 全部
    python review/check_deck.py 2 3        # 只看 slide 2、3
    python review/check_deck.py --overlap  # 只报重叠
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WS = ROOT / "ppt_workspace"
RENDERS = WS / "renders"

# 定稿的每页点击数（P01–P21 合计 71）；22、23 为备份页 B1、B2
EXPECTED_CLICKS = {
    1: 1, 2: 4, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 3, 9: 3, 10: 4, 11: 4,
    12: 4, 13: 4, 14: 6, 15: 3, 16: 5, 17: 4, 18: 3, 19: 4, 20: 2, 21: 0,
    22: 0, 23: 0,
}

# 画布 1280px 宽 = 13.333in = 960pt  →  1px = 0.75pt
PT_PER_PX = 0.75
MIN_PT = 20.0            # 正文下限
MIN_PT_CORNER = 12.0     # 页脚导航、页码、角落括注的绝对下限

# 允许小于 MIN_PT 的元素（角落对照、页码、页脚导航）
SMALL_OK = re.compile(r"页码|pagenum|footer|nav|corner|note|caption", re.I)

FORBIDDEN = [
    (r"每个任务的\s*II", "应统一叫“占机时长”"),
    (r"同一时刻只能干一件活", "P14 已改为“占用区间不能冲突”"),
    (r"(无法|没法)直接求", "P15 已改为“直接求解通常更难”"),
    (r"(可以|能)降到\s*1053", "P19 应写“II ≥ 1053；若达到这个理想节拍…”"),
    (r"每台机器都在忙", "P10 已改为“瓶颈机器尽量连续工作”"),
    (r"等待时长\s*[≥>=]+\s*占机时长", "该泛化已在 P14 删除"),
]

PAGE_LIMITED = [
    (r"\bMFU\b", {3, 19}, "MFU 只在 P03 点一下、P19 标“非实测 MFU”"),
    (r"MTE[123]|Fixpipe", {18}, "硬件缩写只在 P18 角落括注"),
    (r"启动间隔|(?<![A-Za-z])II(?![A-Za-z])", set(range(10, 24)),
     "II 不得在 P10 之前出现"),
]

NEEDS_LABEL = [
    (18, r"\b(518|520|735|311|171|318)\b", r"参考材料|耗时估计",
     "P18 耗时需标“参考材料中的耗时估计”"),
    (19, r"9[89](\.\d+)?\s*%", r"理论占用|非实测",
     "P19 的 99% 需标“理论占用，非实测 MFU”"),
    (19, r"4\.2\d?", r"上限|理想",
     "P19 的 4.2 倍需写明是上限 / 理想值"),
]

HALFWIDTH_Q = re.compile(r"[一-鿿][^\n]{0,40}?\?")


def load(slide: int):
    mf = WS / f"manifest-{slide:02d}.json"
    lay = RENDERS / f"slide-{slide}.layout.json"
    m = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else None
    l = json.loads(lay.read_text(encoding="utf-8")) if lay.exists() else None
    return m, l


def rect(el):
    p = el.get("position") or {}
    return (p.get("left", 0), p.get("top", 0),
            p.get("width", 0), p.get("height", 0))


def overlaps(a, b, tol=4):
    ax, ay, aw, ah = rect(a)
    bx, by, bw, bh = rect(b)
    ix = min(ax + aw, bx + bw) - max(ax, bx)
    iy = min(ay + ah, by + bh) - max(ay, by)
    if ix <= tol or iy <= tol:
        return 0.0
    inter = ix * iy
    smaller = min(aw * ah, bw * bh) or 1
    return inter / smaller


def click_of(name: str) -> int:
    m = re.search(r"_C(\d+)_", name or "")
    return int(m.group(1)) if m else 0


def check(slide: int, only_overlap: bool = False):
    out: list[tuple[str, str]] = []
    m, l = load(slide)

    if m is None:
        out.append(("ERROR", "缺少 manifest"))
        return out
    exp = EXPECTED_CLICKS.get(slide)

    if not only_overlap:
        if exp is not None and m.get("clicks") != exp:
            out.append(("ERROR", f"点击数 {m.get('clicks')}，定稿为 {exp}"))
        used = sorted({a["click"] for a in m.get("animations", [])})
        if used != list(range(1, (m.get("clicks") or 0) + 1)):
            out.append(("ERROR", f"点击编号不连续：{used}"))
        # 每个 k>0 的元素都要有动画
        named = {a["name"] for a in m.get("animations", [])}
        missing = [i["name"] for i in m.get("items", [])
                   if i.get("k", 0) > 0 and i["name"] not in named]
        if missing:
            out.append(("ERROR",
                        f"{len(missing)} 个元素标了点击却无动画，例如 {missing[:3]}"))

    if l is None:
        if not only_overlap:
            out.append(("WARN", "尚未渲染（无 layout.json），仅校验了 manifest"))
        return out

    els = [e for e in l.get("elements", []) if e.get("kind") == "textbox"]
    alltext = " ".join(e.get("text", "") or "" for e in els)

    # 重叠：只比同一点击阶段或更早的元素（后出现的会盖住先出现的，属正常设计的除外）
    for i, a in enumerate(els):
        for b in els[i + 1:]:
            ta, tb = (a.get("text") or "").strip(), (b.get("text") or "").strip()
            if not ta or not tb:
                continue
            ratio = overlaps(a, b)
            if ratio > 0.25:
                ca, cb = click_of(a.get("name", "")), click_of(b.get("name", ""))
                # 文字压文字不是有意设计，一律 ERROR
                lvl = "ERROR"
                out.append((lvl,
                            f"文字重叠 {ratio:.0%}（C{ca}/C{cb}）："
                            f"“{ta[:22]}” × “{tb[:22]}”"))

    if only_overlap:
        return out

    # 字号
    for e in els:
        fs = (e.get("style") or {}).get("fontSize")
        if not fs:
            continue
        pt = fs * PT_PER_PX
        txt = (e.get("text") or "").strip()
        if not txt:
            continue
        top = (e.get("position") or {}).get("top", 0)
        is_footer = top >= 660          # 页脚导航带与页码，无需后排阅读
        floor = MIN_PT_CORNER if (is_footer or SMALL_OK.search(e.get("name", ""))) else MIN_PT
        if pt < floor:
            out.append(("ERROR" if pt < MIN_PT_CORNER else "WARN",
                        f"字号 {fs}px ≈ {pt:.1f}pt < {floor:.0f}pt — “{txt[:24]}”"))

    # 字体
    faces = {(e.get("style") or {}).get("typeface") for e in els}
    faces.discard(None)
    if faces - {"Microsoft YaHei", "微软雅黑"}:
        out.append(("WARN", f"出现非微软雅黑字体：{sorted(faces)}"))

    # 红线
    for pat, why in FORBIDDEN:
        if re.search(pat, alltext):
            out.append(("ERROR", f"红线表述 /{pat}/ — {why}"))

    for pat, allowed, why in PAGE_LIMITED:
        if slide not in allowed:
            hit = re.search(pat, alltext)
            if hit:
                out.append(("ERROR", f"“{hit.group(0)}”不应出现在 slide {slide} — {why}"))

    for pg, numpat, labelpat, why in NEEDS_LABEL:
        if slide == pg and re.search(numpat, alltext) and not re.search(labelpat, alltext):
            out.append(("ERROR", why))

    # 半角问号
    for e in els:
        t = (e.get("text") or "").strip()
        if HALFWIDTH_Q.search(t):
            out.append(("ERROR", f"中文后用了半角问号 — “{t[:28]}”"))

    if not out:
        out.append(("INFO", f"{len(els)} 个文本框，检查通过"))
    return out


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = [a for a in sys.argv[1:]]
    only_overlap = "--overlap" in args
    nums = [int(a) for a in args if a.isdigit()] or sorted(EXPECTED_CLICKS)

    e = w = 0
    for n in nums:
        res = check(n, only_overlap)
        shown = [r for r in res if r[0] != "INFO"] or res
        if only_overlap and not [r for r in res if r[0] != "INFO"]:
            continue
        print(f"\n=== slide {n}（P{n:02d}，定稿 {EXPECTED_CLICKS.get(n, '?')} 次点击）===")
        for lvl, msg in shown:
            print(f"  [{lvl}] {msg}")
            e += lvl == "ERROR"
            w += lvl == "WARN"
    print(f"\n合计 ERROR {e} 条，WARN {w} 条。P01–P21 点击总数应为 71。")
    return 1 if e else 0


if __name__ == "__main__":
    raise SystemExit(main())
