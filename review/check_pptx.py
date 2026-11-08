#!/usr/bin/env python
"""直接校验交付物 .pptx（只读）。比 renders/ 更权威——渲染图可能是旧的。

用法:
    python review/check_pptx.py [路径]     默认 ppt_workspace/animated-draft.pptx
"""

from __future__ import annotations

import collections
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "ppt_workspace" / "animated-draft.pptx"

EXPECTED_CLICKS = {
    1: 1, 2: 4, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 3, 9: 3, 10: 4, 11: 4,
    12: 4, 13: 4, 14: 6, 15: 3, 16: 5, 17: 4, 18: 3, 19: 4, 20: 2, 21: 0,
    22: 0, 23: 0,
}

# 分档字号标准（讨论后的可执行版本）
TIER_BODY = 20.0     # 结论句、正文、过渡语、徽章
TIER_LABEL = 16.0    # 图内色块标签、泳道名
TIER_MICRO = 12.0    # 页脚导航、页码、角落括注

# 判为"图内小标签"的特征：短、且常在色块里
LABEL_MAXLEN = 14

FORBIDDEN = [
    (r"每个任务的\s*II", "应统一叫“占机时长”"),
    (r"同一时刻只能干一件活", "P14 应为“占用区间不能冲突”"),
    (r"(无法|没法)直接求", "P15 应为“直接求解通常更难”"),
    (r"(可以|能)降到\s*1053", "P19 应写“II ≥ 1053…”"),
    (r"每台机器都在忙", "P10 应为“瓶颈机器尽量连续工作”"),
    (r"等待时长\s*[≥>=]+\s*占机时长", "该泛化已删除"),
]
PAGE_LIMITED = [
    (r"\bMFU\b", {3, 19}, "MFU 限 P03 / P19"),
    (r"MTE[123]|Fixpipe", {18}, "硬件缩写限 P18"),
    (r"启动间隔|(?<![A-Za-z])II(?![A-Za-z])", set(range(10, 24)), "II 不得早于 P10"),
]


def runs(xml: str):
    """产出 (磅值, 文字)。"""
    for m in re.finditer(r"<a:rPr\b[^>]*?sz=\"(\d+)\"[^>]*?>.*?</a:rPr>\s*<a:t>([^<]*)</a:t>"
                         r"|<a:rPr\b[^>]*?sz=\"(\d+)\"[^>]*?/>\s*<a:t>([^<]*)</a:t>", xml, re.S):
        sz = int(m.group(1) or m.group(3))
        yield sz / 100, (m.group(2) or m.group(4) or "")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"找不到 {path}")
        return 2

    z = zipfile.ZipFile(path)
    err = warn = 0

    # --- 结构 ---
    pres = z.read("ppt/presentation.xml").decode("utf-8", "replace")
    m = re.search(r'sldSz[^/]*cx="(\d+)"[^/]*cy="(\d+)"', pres)
    if m:
        cx, cy = int(m.group(1)), int(m.group(2))
        print(f"幻灯片 {cx / 914400:.3f} × {cy / 914400:.3f} 英寸，比例 {cx / cy:.4f}")
        if abs(cx / cy - 16 / 9) > 0.002:
            print("  [ERROR] 不是 16:9")
            err += 1
    bad_zip = z.testzip()
    print(f"zip 自检 {'OK' if not bad_zip else bad_zip}；"
          f"嵌入图片 {len([n for n in z.namelist() if n.startswith('ppt/media/') and '.' in n])} 张")

    total_click = 0
    size_tally: collections.Counter[float] = collections.Counter()
    problems: list[str] = []

    for i in sorted(EXPECTED_CLICKS):
        name = f"ppt/slides/slide{i}.xml"
        if name not in z.namelist():
            problems.append(f"slide{i}: 缺失")
            err += 1
            continue
        x = z.read(name).decode("utf-8", "replace")
        text = " ".join(re.findall(r"<a:t>([^<]*)</a:t>", x))

        clicks = len(re.findall(r'nodeType="clickEffect"', x))
        if i <= 21:
            total_click += clicks
        if clicks != EXPECTED_CLICKS[i]:
            problems.append(f"slide{i}: clickEffect {clicks}，定稿 {EXPECTED_CLICKS[i]}")
            err += 1

        ids = set(re.findall(r'<p:cNvPr\s+id="(\d+)"', x))
        dangling = set(re.findall(r'spid="(\d+)"', x)) - ids
        if dangling:
            problems.append(f"slide{i}: 动画指向不存在的形状 {sorted(dangling)[:4]}")
            err += 1

        for pat, why in FORBIDDEN:
            if re.search(pat, text):
                problems.append(f"slide{i}: 红线表述 — {why}")
                err += 1
        for pat, allowed, why in PAGE_LIMITED:
            if i not in allowed:
                hit = re.search(pat, text)
                if hit:
                    problems.append(f"slide{i}: 出现“{hit.group(0)}” — {why}")
                    err += 1
        for hit in re.finditer(r"[一-鿿][^\s]{0,30}?\?", text):
            problems.append(f"slide{i}: 中文后半角问号 — {hit.group(0)[:24]}")
            err += 1

        for pt, t in runs(x):
            t = t.strip()
            if not t:
                continue
            size_tally[pt] += 1
            floor = TIER_LABEL if len(t) <= LABEL_MAXLEN else TIER_BODY
            if pt < TIER_MICRO:
                problems.append(f"slide{i}: {pt}pt 低于最低档 — “{t[:20]}”")
                err += 1
            elif pt < floor and pt >= TIER_MICRO:
                warn += 1

    print(f"\nP01–P21 clickEffect 合计 {total_click}（应为 71）"
          f" {'OK' if total_click == 71 else 'MISMATCH'}")

    under_body = sum(n for pt, n in size_tally.items() if pt < TIER_BODY)
    under_label = sum(n for pt, n in size_tally.items() if pt < TIER_LABEL)
    print(f"字号：共 {sum(size_tally.values())} 个文字块，"
          f"< {TIER_BODY:.0f}pt 有 {under_body} 个，< {TIER_LABEL:.0f}pt 有 {under_label} 个")
    print("  分布 " + "，".join(f"{pt:g}pt×{n}" for pt, n in sorted(size_tally.items())[:12]))

    if problems:
        print(f"\n问题 {len(problems)} 条：")
        for p in problems[:40]:
            print("  " + p)
        if len(problems) > 40:
            print(f"  …… 另有 {len(problems) - 40} 条")
    else:
        print("\n结构、点击分组、术语红线：全部通过。")

    print(f"\nERROR {err} 条，字号偏小 WARN {warn} 条。")
    return 1 if err else 0


if __name__ == "__main__":
    raise SystemExit(main())
