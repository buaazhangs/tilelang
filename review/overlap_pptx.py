#!/usr/bin/env python
"""检测 .pptx 里文字框的几何重叠（只读）。

会考虑"遮挡"：若两段文字之间的 z 序上存在一个不透明形状，且它完全盖住
下面那段文字，则判为"原地替换"而非缺陷（GPT 的 clear_old_* 做法）。

用法:
    python review/overlap_pptx.py              # 默认扫 PPT交付/ 下的 pptx
    python review/overlap_pptx.py --all        # 扫 ppt_workspace/ 与 PPT交付/
"""

from __future__ import annotations

import glob
import os
import re
import sys
import zipfile

EMU = 914400.0
SP = re.compile(r"<p:sp>.*?</p:sp>", re.S)
OFF = re.compile(r'<a:off x="(-?\d+)" y="(-?\d+)"/>\s*<a:ext cx="(\d+)" cy="(\d+)"/>')


def shape_fill(sp: str) -> str | None:
    """只取 <p:spPr> 内的填充，避免把文字颜色当成形状填充。"""
    m = re.search(r"<p:spPr[^>]*>(.*?)</p:spPr>", sp, re.S)
    if not m:
        return None
    # 先剥掉 <a:ln>…</a:ln>：边框的 noFill 不代表形状没有填充
    body = re.sub(r"<a:ln\b[^>]*>.*?</a:ln>", "", m.group(1), flags=re.S)
    body = re.sub(r"<a:ln\b[^>]*/>", "", body)
    f = re.search(r'<a:solidFill[^>]*>\s*<a:srgbClr[^>]*val="(\w+)"', body)
    if f:
        return f.group(1)
    return None if "<a:noFill/>" in body else None


def covers(outer, inner, pad=0.0) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return (ox - pad <= ix and oy - pad <= iy
            and ox + ow + pad >= ix + iw and oy + oh + pad >= iy + ih)


def scan(path: str):
    z = zipfile.ZipFile(path)
    out = []
    for i in range(1, 40):
        name = f"ppt/slides/slide{i}.xml"
        if name not in z.namelist():
            continue
        xml = z.read(name).decode("utf-8", "replace")
        shapes = []
        for order, sp in enumerate(SP.findall(xml)):
            m = OFF.search(sp)
            if not m:
                continue
            rect = tuple(int(v) / EMU for v in m.groups())
            text = "".join(re.findall(r"<a:t>([^<]*)</a:t>", sp)).strip()
            shapes.append({"z": order, "rect": rect, "text": text,
                           "fill": shape_fill(sp)})

        texts = [s for s in shapes if s["text"]]
        for a in range(len(texts)):
            for b in range(a + 1, len(texts)):
                s1, s2 = texts[a], texts[b]
                (x1, y1, w1, h1), (x2, y2, w2, h2) = s1["rect"], s2["rect"]
                ox = min(x1 + w1, x2 + w2) - max(x1, x2)
                oy = min(y1 + h1, y2 + h2) - max(y1, y2)
                if ox <= 0 or oy <= 0:
                    continue
                frac = (ox * oy) / max(min(w1 * h1, w2 * h2), 1e-9)
                if frac <= 0.3:
                    continue

                lower, upper = (s1, s2) if s1["z"] < s2["z"] else (s2, s1)
                # 1) 上面那个本身不透明且完全盖住下面的 → 原地替换
                hidden_by = None
                if upper["fill"] and covers(upper["rect"], lower["rect"]):
                    hidden_by = "上层形状本身"
                else:
                    # 2) 两者之间存在不透明遮挡块
                    for s in shapes:
                        if (lower["z"] < s["z"] < upper["z"] and s["fill"]
                                and not s["text"] and covers(s["rect"], lower["rect"])):
                            hidden_by = "遮挡块"
                            break
                out.append({
                    "slide": i, "frac": frac, "ok": hidden_by is not None,
                    "why": hidden_by,
                    "t1": lower["text"][:24], "t2": upper["text"][:24],
                    "y": lower["rect"][1],
                })
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    paths = sorted(glob.glob("PPT*/*.pptx"))
    if "--all" in sys.argv:
        paths = sorted(glob.glob("ppt_workspace/*.pptx")) + paths
    paths = [p for p in paths if not os.path.basename(p).startswith("~$")]
    if not paths:
        print("找不到 pptx")
        return 2

    bad_total = 0
    for p in paths:
        try:
            res = scan(p)
        except Exception as exc:
            print(f"{p}: 读取失败 {exc}")
            continue
        real = [r for r in res if not r["ok"]]
        okd = [r for r in res if r["ok"]]
        bad_total += len(real)
        print(f"\n### {p}")
        print(f"    需处理 {len(real)} 处；原地替换（已遮挡，无需处理）{len(okd)} 处")
        for r in real:
            print(f"    [需改] slide{r['slide']:2d} {r['frac']:3.0%} y={r['y']:.2f}in"
                  f'  "{r["t1"]}" × "{r["t2"]}"')
        for r in okd:
            print(f"    [正常] slide{r['slide']:2d} {r['frac']:3.0%} 由{r['why']}遮挡"
                  f'  "{r["t1"]}" → "{r["t2"]}"')
    return 1 if bad_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
