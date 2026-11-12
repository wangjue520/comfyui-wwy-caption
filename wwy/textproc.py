# -*- coding: utf-8 -*-
"""标签筛选、提示词合并、AI 优化提示词的指令。"""
import fnmatch
import json
import os
import re
import threading

from . import locate

# ---------------- 标签 ----------------
RATING = {"general", "sensitive", "questionable", "explicit", "safe", "nsfw"}
# 人物固有特征（换角色时要去掉的）：发色发型、瞳色、兽耳尾巴角、体型肤色……
_TRAIT_PATTERNS = [
    r".*_hair$", r"^hair_.*", r".*_eyes$", r".*_pupils$", r".*bangs$", r"^sidelocks$", r"^ahoge$", r".*ponytail$",
    r"^twintails$", r".*braids?$", r".*hair_bun$", r"^double_bun$", r".*_ears$", r".*_tail$", r".*horns?$", r".*_wings$",
    r"^wings$", r"^halo$", r"^heterochromia$", r".*_breasts$", r"^breasts$", r".*_skin$", r"^dark-skinned_.*",
    r"^mole.*", r"^freckles$", r"^fang$", r"^skin_fang$", r".*_eyebrows$", r"^eyelashes$", r"^flat_chest$",
    r"^(bob|hime|pixie|bowl|buzz)_cut$", r"^undercut$", r"^twin_drills$", r"^official_alternate_hair.*",
]
_TRAIT_RE = re.compile("|".join(f"(?:{p})" for p in _TRAIT_PATTERNS))

_dict = None
_dict_lock = threading.Lock()


def tag_categories():
    """字典插件（comfyui-zh-tag-autocomplete）的 tags.json：标签 → 类别（0 通用 1 画师 3 作品 4 角色 5 元标签）。"""
    global _dict
    with _dict_lock:
        if _dict is None:
            _dict = {}
            p = os.path.join(locate.sibling_plugin("comfyui-zh-tag-autocomplete"), "data", "tags.json")
            try:
                with open(p, encoding="utf-8") as f:
                    for row in json.load(f):
                        _dict[row[0]] = int(row[1])
            except (OSError, ValueError, IndexError, TypeError):
                pass
        return _dict


def split(text):
    return [t.strip() for t in re.split(r"[,\n，]", text or "") if t.strip()]


def norm(tag):
    """比较用：小写、去掉转义和权重括号、空格换下划线。"""
    t = tag.strip().lower()
    m = re.fullmatch(r"\((.*):[\d.]+\)", t)
    if m:
        t = m.group(1)
    t = t.replace("\\(", "(").replace("\\)", ")").strip()
    return re.sub(r"\s+", "_", t)


def _patterns(text):
    return [norm(p) for p in split(text)]


def _match(n, pats):
    return any(fnmatch.fnmatchcase(n, p) for p in pats)


def filter_tags(tags, drop_artist=True, drop_character=False, drop_copyright=False, drop_meta=True, drop_traits=False,
                blacklist="", whitelist="", max_tags=0):
    cats = tag_categories()
    black, white = _patterns(blacklist), _patterns(whitelist)
    out, seen = [], set()
    for t in split(tags):
        n = norm(t)
        if n in seen:
            continue
        seen.add(n)
        c = cats.get(n)
        if drop_meta and (n in RATING or n.startswith("rating:") or c == 5):
            continue
        if drop_artist and c == 1:
            continue
        if drop_character and c == 4:
            continue
        if drop_copyright and c == 3:
            continue
        if drop_traits and _TRAIT_RE.fullmatch(n):
            continue
        if black and _match(n, black):
            continue
        if white and not _match(n, white):
            continue
        out.append(t)
    if max_tags and len(out) > max_tags:
        out = out[:max_tags]
    return ", ".join(out)


def escape_parens(tag):
    return re.sub(r"(?<!\\)([()])", r"\\\1", tag)


# ---------------- 合并 ----------------
def merge(prompt, tags="", text="", dedupe=True, tags_first=False):
    """主提示词 + 反推标签 + 描述句子。主提示词里已经有的标签不重复加。"""
    main = (prompt or "").strip().rstrip(",").strip()
    have = {norm(t) for t in split(main)} if dedupe else set()
    add = []
    for t in split(tags):
        n = norm(t)
        if dedupe and n in have:
            continue
        have.add(n)
        add.append(t)
    parts = [", ".join(add), main] if tags_first else [main, ", ".join(add)]
    out = ", ".join(p for p in parts if p)
    text = (text or "").strip()
    if text:
        out = (out.rstrip(".").rstrip(",") + ". " + text) if out else text
    return out


# ---------------- AI 优化：写法和条件在 enhance.py ----------------


# AI 写出来的画风 / 画质类词（教程规定不带画风；Anima 也不需要这些）
_STYLE_RE = re.compile(r"realis|photo|painterly|watercolor|oil paint|sketch|\b3d\b|render|cinematic|aesthetic|\bstyle|"
                       r"inspired|\bartist|\bby\s|professional|\b[48]k\b|\bhdr\b|unreal|octane|skin texture|"
                       r"chromatic aberration|artstation|trending", re.I)
MAX_TAGS = 50
NO_STYLE_CAPTION = ("Do not describe the art style, medium, rendering technique, line work, shading or overall color palette, "
                    "and do not name any artist or style.")


def drop_style(tags):
    return [t for t in tags if not _STYLE_RE.search(t)]


def clean_prompt(s, tags_mode=False):
    """去掉思考、代码块、「Prompt:」前缀；tags_mode：标签部分去重、括号转义（句子部分不动）。"""
    s = re.sub(r"<think>.*?</think>", "", s or "", flags=re.S).strip()
    s = re.sub(r"^```\w*\s*|\s*```$", "", s).strip()
    s = re.sub(r"^(\*\*)?(prompt|optimized prompt|rewritten prompt|output|example output)(\*\*)?\s*[:：]\s*", "", s, flags=re.I)
    s = re.sub(r"\s*\n+\s*", " ", s).strip().strip('"').strip()
    if not tags_mode:
        return s
    m = re.search(r"\.\s+(?=[A-Z])", s)   # 标签和句子之间的句号
    head, tail = (s[:m.start()], s[m.end():]) if m else (s, "")
    tags, seen = [], set()
    for t in split(head):
        n = norm(t)
        if n and n not in seen:
            seen.add(n)
            tags.append(t if r"\(" in t else escape_parens(t) if re.fullmatch(r"[^()]*\([^():]*\)[^()]*", t) else t)
    out = ", ".join(drop_style(tags)[:MAX_TAGS])
    return (out + ". " + tail.strip()) if tail.strip() else out
