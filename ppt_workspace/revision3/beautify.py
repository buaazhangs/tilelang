# -*- coding: utf-8 -*-
"""
三章重构版 -> 视觉优化版

只改「表现层」：字体族、文字颜色、字重、字距、形状圆角、装饰性版式骨架。
严格不动：任何 <a:t> 文本、形状 id 与坐标、动画 XML、备注、图表、媒体、页数。

做法：逐字节复制原 pptx 的所有部件，只对 ppt/slides/slideN.xml 做
「按 <p:sp> 块切分 -> 按角色重映射样式 -> 拼回 -> 插入装饰形状」的字符串级改写，
因此 WPS 生成的动画引用、命名空间、扩展节点全部原样保留。
"""
import re
import sys
import zipfile

SRC = r'D:\Dev\tilelang\PPT交付\TileLang_昇腾模调度_三章重构版.pptx'
DST = r'D:\Dev\tilelang\PPT交付\TileLang_昇腾模调度_三章重构版_视觉优化版.pptx'

ANS = 'http://schemas.openxmlformats.org/drawingml/2006/main'

# ---- 新色板：深墨技术稿 + 单点红强调 ------------------------------------
INK    = '16202B'   # 标题
BODY   = '1B2430'   # 正文深
LEAD   = '3E4A57'   # 导语 / 次级标题
MUTED  = '6E7B89'   # 次要说明
FAINT  = 'AAB4BF'   # 页脚
HINT   = '8A97A4'   # 过渡语
LINE   = 'E4EAF0'   # 分隔线 / 描边
CARD   = 'F7F9FC'   # 内容分组底
SECNUM = 'DCE4EC'   # 章节页大数字（略深，保证序号可辨）

SEMANTIC = ('326DA8', 'C16B24', '28795F', '7352A0', 'FFFFFF')  # 语义色，绝不改

SECTION_PAGES = {2, 6, 12}
COVER, END = 1, 24

SP_RE = re.compile(r'<p:sp>.*?</p:sp>', re.S)


def info(block):
    m = re.search(r'<p:cNvPr id="(\d+)"', block)
    sid = m.group(1) if m else ''
    off = re.search(r'<a:off x="(-?\d+)" y="(-?\d+)"', block)
    x = int(off.group(1)) if off else 0
    y = int(off.group(2)) if off else 0
    text = ''.join(re.findall(r'<a:t>(.*?)</a:t>', block, re.S))
    szs = [int(v) for v in re.findall(r'\bsz="(\d+)"', block)]
    return sid, x, y, text, (max(szs) if szs else 0)


def role(page, block, sid, x, y, text, sz):
    if page in (COVER, END):
        return 'special'
    if page in SECTION_PAGES:
        if sz >= 8000:
            return 'section_num'
        if sz >= 4000:
            return 'section_title'
        if text and y > 4800000:
            return 'section_question'
        return 'section_other'
    if not text.strip():
        if y < 1300000 and 'B3202B' in block:
            return 'top_rule'
        return 'decor'
    if y < 700000 and sz >= 3000:
        return 'title'
    if y >= 6450000:
        return 'footer'
    if 6050000 <= y < 6450000:
        return 'transition'
    if 1300000 <= y < 1750000:
        # 「已知 / 未知 / 目标」是语义色标签，保留原色
        if text.startswith(('已知：', '未知：', '目标：')):
            return 'semantic_lead'
        return 'lead'
    if sz >= 2400 and ('B3202B' in block or 'C16B24' in block):
        return 'emphasis'          # 关键数字与结论：保留红 / 橙
    return 'content'


def remap_color(block, r):
    """按角色重映射文字颜色。语义色与白色一律不动。"""
    def sub(src, dst):
        nonlocal block
        block = block.replace('val="%s"' % src, 'val="%s"' % dst)

    if r in ('title', 'section_title'):
        sub('B3202B', INK)
    elif r == 'lead':
        sub('B3202B', LEAD)
    elif r == 'transition':
        sub('B3202B', HINT)
    elif r == 'footer':
        sub('66707B', FAINT)
    elif r == 'section_num':
        sub('F9E9EB', SECNUM)
    elif r == 'section_question':
        sub('242933', BODY)
    elif r == 'top_rule':
        sub('B3202B', LINE)        # 长线转浅灰，红只留左端一小段
    elif r == 'content':
        sub('B3202B', LEAD)        # 正文里的小标签不再用红，红只留给结论
    elif r in ('emphasis', 'semantic_lead', 'special', 'section_other', 'decor'):
        pass                       # 强调色 / 语义色 / 装饰红 全部保留
    else:
        pass

    # 全角色通用的中性色调整（不影响语义色、白色、红色强调）
    if r != 'footer':
        sub('242933', BODY)
        sub('66707B', MUTED)
    sub('DCE1E6', LINE)
    return block


def apply_font(block, r, page):
    """中文保留微软雅黑（零掉字风险），西文/数字换 Segoe UI；代码块换 Consolas。"""
    if r == 'section_num':
        latin = cs = 'Segoe UI Light'
    elif page == 13 and ('T.Pipelined' in block or 'for kv in' in block):
        latin = cs = 'Consolas'    # 代码：等宽，技术感
    else:
        latin = cs = 'Segoe UI'
    block = block.replace('<a:latin typeface="Microsoft YaHei"/>',
                          '<a:latin typeface="%s"/>' % latin)
    block = block.replace('<a:cs typeface="Microsoft YaHei"/>',
                          '<a:cs typeface="%s"/>' % cs)
    return block                  # <a:ea> 保持中文雅黑


def round_tasks(block, r):
    """任务色块由直角改圆角（纯视觉，不动坐标）。"""
    if r != 'decor' or not any('val="%s"' % c in block for c in SEMANTIC[:3]):
        return block
    return re.sub(
        r'(<a:prstGeom[^>]*?\bprst=")rect("[^>]*>)<a:avLst\s*/>',
        r'\1roundRect\2<a:avLst><a:gd name="adj" fmla="val 14000"/></a:avLst>',
        block)


def typography(block, r):
    if r == 'title':
        # 标题加一点字距；导语去粗，拉开「标题 / 导语 / 正文」三级
        block = re.sub(r'(<a:(?:rPr|defRPr) sz="3150" b="1")',
                       r'\1 spc="60"', block)
    elif r == 'section_title':
        block = re.sub(r'(<a:(?:rPr|defRPr) sz="4800" b="1")',
                       r'\1 spc="80"', block)
    elif r == 'lead':
        block = re.sub(r'(<a:(?:rPr|defRPr) sz="2175" b=")1(")',
                       r'\g<1>0\2', block)
    elif r == 'transition':
        block = block.replace('sz="1800"', 'sz="1650"')
        block = re.sub(r'(<a:(?:rPr|defRPr) sz="1800" b=")1(")',
                       r'\g<1>0\2', block)
    return block


# ---- 装饰形状 ------------------------------------------------------------
def deco_sp(sid, name, x, y, w, h, fill, prst='rect', adj=None, line=None):
    av = '<a:avLst/>' if adj is None else (
        '<a:avLst><a:gd name="adj" fmla="val %d"/></a:avLst>' % adj)
    fill_xml = ('<a:solidFill><a:srgbClr val="%s"/></a:solidFill>' % fill
                if fill else '<a:noFill/>')
    ln_xml = ('<a:ln w="9525"><a:solidFill><a:srgbClr val="%s"/>'
              '</a:solidFill></a:ln>' % line if line else '<a:ln><a:noFill/></a:ln>')
    return (
        '<p:sp xmlns:a="%s"><p:nvSpPr><p:cNvPr id="%d" name="%s"/>'
        '<p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr>'
        '<a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
        '<a:prstGeom prst="%s">%s</a:prstGeom>%s%s</p:spPr>'
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody></p:sp>'
        % (ANS, sid, name, x, y, w, h, prst, av, fill_xml, ln_xml))


def decorations(page):
    """返回 (底层形状, 顶层形状)。全部为纯几何，不含任何文字。"""
    bottom, top, uid = [], [], 9000 + page * 20

    if page not in (COVER, END) and page not in SECTION_PAGES:
        # 内容页：一张极浅圆角分组卡，把标题与页脚留在白底上
        # 卡片左右各留 24px 内边距，内容不再贴边
        bottom.append(deco_sp(uid, 'BEAUTY_card', 381000, 1295400,
                              11430000, 4860000, CARD, 'roundRect', 2200))
        # 标题左侧红色竖标（与标题首字留 12px）+ 长线左端的一小段红
        top.append(deco_sp(uid + 1, 'BEAUTY_title_mark', 457200, 400050,
                           38100, 438150, 'B3202B', 'roundRect', 30000))
        top.append(deco_sp(uid + 2, 'BEAUTY_rule_accent', 609600, 1181100,
                           914400, 25400, 'B3202B', 'roundRect', 40000))

    if page in SECTION_PAGES:
        # 三张章节页各给一个不同的几何钩子，破除「占位页」感
        bx, by = 8001000, 1600200
        if page == 2:      # 串行：三段分离的横条
            for k, (yy, col, ww) in enumerate([
                    (by, '326DA8', 1200000), (by + 900000, 'C16B24', 1200000),
                    (by + 1800000, '28795F', 1200000)]):
                top.append(deco_sp(uid + k, 'BEAUTY_sec', bx, yy, ww, 300000,
                                   col, 'roundRect', 16000))
                top.append(deco_sp(uid + 10 + k, 'BEAUTY_sec_track',
                                   bx + ww + 180000, yy, 1800000, 300000,
                                   LINE, 'roundRect', 16000))
        elif page == 6:    # Scalar -> 三个单元
            top.append(deco_sp(uid, 'BEAUTY_sec', bx, by, 3300000, 380000,
                               'B3202B', 'roundRect', 12000))
            for k, col in enumerate(['326DA8', 'C16B24', '28795F']):
                top.append(deco_sp(uid + 1 + k, 'BEAUTY_sec',
                                   bx + k * 1150000, by + 1500000, 950000, 380000,
                                   col, 'roundRect', 12000))
        else:              # 三组约束方块
            for k, col in enumerate(['326DA8', '7352A0', 'C16B24']):
                top.append(deco_sp(uid + k, 'BEAUTY_sec',
                                   bx + k * 1150000, by + 900000, 950000, 950000,
                                   col, 'roundRect', 18000))
    return ''.join(bottom), ''.join(top)


def transform(xml, page):
    out, last = [], 0
    for m in SP_RE.finditer(xml):
        out.append(xml[last:m.start()])
        block = m.group(0)
        sid, x, y, text, sz = info(block)
        r = role(page, block, sid, x, y, text, sz)
        block = remap_color(block, r)
        block = typography(block, r)
        block = round_tasks(block, r)
        block = apply_font(block, r, page)
        out.append(block)
        last = m.end()
    out.append(xml[last:])
    s = ''.join(out)

    bottom, top = decorations(page)
    if bottom:
        s = s.replace('</p:grpSpPr>', '</p:grpSpPr>' + bottom, 1)
    if top:
        s = s.replace('</p:spTree>', top + '</p:spTree>', 1)
    return s


def main():
    counts = {}
    with zipfile.ZipFile(SRC, 'r') as zin, zipfile.ZipFile(DST, 'w') as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            m = re.match(r'ppt/slides/slide(\d+)\.xml$', item.filename)
            if m:
                page = int(m.group(1))
                before = data.decode('utf-8')
                after = transform(before, page)
                data = after.encode('utf-8')
                counts[page] = (len(before), len(after))
            zout.writestr(item, data)
    print('written:', DST)
    print('pages transformed:', len(counts))
    bad = [p for p, (a, b) in counts.items() if b == 0]
    print('empty outputs:', bad)


if __name__ == '__main__':
    main()
