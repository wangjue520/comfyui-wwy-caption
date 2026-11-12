# -*- coding: utf-8 -*-
"""
AI 优化提示词：写法、条件开关、查字典。

写法参考 comfyui-mcp 的 anima-base / krea2-txt2img 两个 skill：
  · Anima：质量词 → 人数和主体标签 → 内容标签 → 2 到 4 句英文描述；提示词太短太空时容易画出不想要的东西。
  · Krea 2：用视觉语言模型（Qwen3-VL）读提示词，写成完整、具体的英文段落，像在描述一张已经存在的图。
条件开关：关掉的那一类（外貌、服装、动作表情、背景、光线构图）AI 不许新加；用户自己写的照样保留。
查字典（Anima 写法）：用中文标签补全插件的字典（comfyui-zh-tag-autocomplete/data/tags.json），
  写之前把用户写的中文查成标准标签交给 AI；写完逐个核对标签，字典里没有的（编出来的）和画师标签删掉。
"""
import json
import os
import re
import threading

from . import locate
from . import textproc as tp

# ---------------- 写法 ----------------
NO_STYLE = ("Never add artist names, '@' mentions, art-style or medium tags (such as 'by xxx', 'watercolor', 'sketch', "
            "'realistic'), and never add tags or sentences the user did not ask for that change the rating.")

EXAMPLE_IN = "masterpiece, best quality, 1girl, 图书馆里看书的女孩"
EXAMPLE_ANIMA = ("masterpiece, best quality, 1girl, solo, sitting, reading, holding book, open book, library, bookshelf, "
                 "indoors, school uniform, long hair, looking down, smile, window, sunlight, upper body. "
                 "A girl with long hair sits at a wooden table in a quiet library, reading an open book with a gentle smile. "
                 "She wears a neat school uniform and rests one hand on the page. Tall bookshelves fill the background, and "
                 "warm afternoon sunlight falls through the window behind her. Upper body shot from a slightly low angle.")
EXAMPLE_NL = ("An anime illustration of a girl with long hair sitting at a wooden table in a quiet library, reading an open "
              "book with a gentle smile. She wears a neat school uniform and rests one hand on the page. Tall bookshelves "
              "fill the background on both sides, and warm afternoon sunlight falls through the tall window behind her. "
              "The shot is an upper body view from a slightly low angle, with the girl in the center of the frame.")

ANIMA = "Anima（标签 + 自然语言）"
NATURAL = "纯自然语言（Krea2 等）"
TRANSLATE = "只翻译整理（不加内容）"
MODES = [ANIMA, NATURAL, TRANSLATE]

_ANIMA_SYS = """You write prompts for Anima, an anime image model that reads Danbooru tags followed by English sentences. Short or vague prompts make it produce unwanted content, so make the scene concrete.
Rewrite the user's prompt (Chinese or English) in this order:
1. The user's quality tags exactly as given (masterpiece, best quality, score_7, year 2025, newest, safe...).
2. Count and subject tags (1girl, 2boys, solo...), then every character and series tag the user gave.
3. Content tags: the user's important tags first, then a few tags that fit the request{allowed}. Use real Danbooru tags only, comma separated, lowercase, spaces instead of underscores (keep tags like score_7 as they are). When dictionary tags are given for the user's Chinese words, use the ones that match the meaning, spelled exactly as given. Drop duplicates and tags that contradict the request. At most 40 tags in total; never pad the list.
4. End the tag list with a period, then write 2 to 4 short English sentences about {sentence_topics}. State what is in the picture as plain facts; no words like seems or might.
{forbid}Never list things that are absent. Never repeat a tag. {no_style} Reply with the prompt only.

Example input: {ex_in}
Example output: {ex_out}"""

_NATURAL_SYS = """You write prompts for an image model that reads natural English through a vision-language encoder (Krea 2). It follows long, specific descriptions much better than keyword lists.
Rewrite the user's prompt (Chinese or English, maybe Danbooru tags) into one English paragraph of 4 to 6 sentences, written as if describing a picture that already exists: the main subject and what they are doing first, then {sentence_topics}. Use concrete nouns and verbs, no vague praise such as beautiful or stunning, no words like seems or might. Keep every subject and intention from the user. Say it is an anime illustration when the request is anime-like.
{forbid}Never describe things that are absent. {no_style} Reply with the paragraph only.

Example input: {ex_in}
Example output: {ex_out}"""

_TRANSLATE_SYS = ("Translate the user's text-to-image prompt into English. Keep Danbooru tags as comma-separated tags (lowercase, "
                  "spaces instead of underscores) and sentences as sentences. When dictionary tags are given for the user's "
                  "Chinese words, use the ones that match the meaning, spelled exactly as given. Do not add or remove content. Reply with the prompt only.")

# ---------------- 条件：AI 可以补哪几类 ----------------
# (开关名, 英文说法, 判断标签属于这一类的规则)
_W = lambda words: re.compile(r"(?:^|_)(?:" + "|".join(words) + r")(?:s|es)?(?:$|_)")
CLOTHES_RE = _W([
    "dress", "skirt", "shirt", "t-shirt", "blouse", "uniform", "jacket", "coat", "hoodie", "sweater", "cardigan", "vest",
    "kimono", "yukata", "hakama", "apron", "bikini", "swimsuit", "swimwear", "leotard", "bodysuit", "pants", "shorts",
    "jeans", "trousers", "thighhigh", "pantyhose", "kneehigh", "sock", "legwear", "footwear", "headwear", "shoe", "boot",
    "sandal", "high_heel", "loafer", "glove", "hat", "cap", "beret", "hood", "necktie", "bowtie", "neckerchief", "scarf",
    "choker", "collar", "ribbon", "hair_ornament", "hairclip", "hair_bow", "hairband", "headband", "earring", "jewelry",
    "necklace", "bracelet", "armor", "cape", "cloak", "robe", "bra", "panties", "underwear", "lingerie", "serafuku",
    "sailor_collar", "suit", "tuxedo", "maid", "camisole", "tank_top", "crop_top", "sleeve", "sleeveless", "frill",
    "lace", "garter", "belt", "glasses", "eyewear", "veil", "tiara", "crown", "clothes", "clothing", "outfit", "costume",
    "nude", "naked", "topless", "barefoot", "bare_shoulders", "midriff", "cleavage", "neck_ribbon", "bow"])
POSE_RE = _W([
    "smile", "grin", "laughing", "blush", "crying", "tears", "open_mouth", "closed_mouth", "closed_eyes", "one_eye_closed",
    "wink", "frown", "pout", "angry", "sad", "surprised", "expressionless", "looking_at_viewer", "looking_away",
    "looking_back", "looking_down", "looking_up", "looking_to_the_side", "sitting", "standing", "lying", "kneeling",
    "squatting", "walking", "running", "jumping", "leaning_forward", "leaning_back", "arms_up", "arms_behind_back",
    "arm_up", "hand_up", "hands_up", "hand_on_hip", "hands_on_hips", "hand_on_own_chest", "hand_on_own_face", "waving",
    "pointing", "crossed_arms", "crossed_legs", "head_tilt", "v", "peace_sign", "outstretched_arm", "reaching",
    "hugging", "stretching", "dancing", "singing", "contrapposto", "parted_lips", "teeth", "tongue_out", ":d", ":o"])
SCENE_RE = _W([
    "background", "indoors", "outdoors", "sky", "cloud", "cloudy_sky", "room", "street", "city", "cityscape", "scenery",
    "forest", "beach", "ocean", "sea", "lake", "river", "school", "classroom", "library", "bedroom", "cafe",
    "train_station", "park", "garden", "field", "mountain", "night", "day", "sunset", "sunrise", "rain", "snow",
    "building", "window", "tree", "grass", "stage", "bookshelf", "bed", "table", "chair", "road", "bridge", "shrine",
    "temple", "alley", "rooftop", "starry_sky", "night_sky", "blue_sky", "water", "puddle", "umbrella_stand"])
LIGHT_RE = _W([
    "lighting", "sunlight", "moonlight", "backlighting", "shadow", "glowing", "lens_flare", "light_rays",
    "light_particles", "sunbeam", "bloom", "depth_of_field", "blurry_background", "bokeh", "upper_body", "lower_body",
    "cowboy_shot", "full_body", "portrait", "close-up", "wide_shot", "very_wide_shot", "from_above", "from_below",
    "from_side", "from_behind", "dutch_angle", "pov", "profile", "feet_out_of_frame", "head_out_of_frame",
    "dynamic_angle", "dappled_sunlight", "rim_lighting", "light", "dark", "silhouette"])

CONDS = [
    ("补外貌", "the character's appearance (hair, eyes, body)", lambda n: bool(tp._TRAIT_RE.fullmatch(n))),
    ("补服装", "clothing and accessories", lambda n: bool(CLOTHES_RE.search(n))),
    ("补动作表情", "pose, action and facial expression", lambda n: bool(POSE_RE.search(n))),
    ("补背景", "background and setting", lambda n: bool(SCENE_RE.search(n))),
    ("补光线构图", "lighting, framing and camera angle", lambda n: bool(LIGHT_RE.search(n))),
]
_SENTENCE = {"补外貌": "appearance", "补服装": "outfit", "补动作表情": "pose and expression",
             "补背景": "background", "补光线构图": "lighting and framing"}


def _topics(on):
    t = ["who the subject is and what they are doing"] + [_SENTENCE[k] for k, _, _ in CONDS if on.get(k, True)]
    return ", ".join(t[:-1]) + " and " + t[-1] if len(t) > 1 else t[0]


def system_prompt(mode, on):
    """on：{开关名: True/False}，没给的当作打开。"""
    off = [desc for k, desc, _ in CONDS if not on.get(k, True)]
    allowed = [desc for k, desc, _ in CONDS if on.get(k, True)]
    forbid = ""
    if off:
        forbid = ("Do not add anything about " + "; ".join(off) + " - no tags and no words in the sentences - unless the "
                  "user wrote it. Keep whatever the user wrote about them.\n")
    if mode == ANIMA:
        return _ANIMA_SYS.format(allowed=(" (" + ", ".join(allowed) + ")") if allowed else " (none: only the user's own content)",
                                 sentence_topics=_topics(on), forbid=forbid, no_style=NO_STYLE,
                                 ex_in=EXAMPLE_IN, ex_out=EXAMPLE_ANIMA)
    if mode == NATURAL:
        return _NATURAL_SYS.format(sentence_topics=_topics(on).replace("who the subject is and what they are doing, ", "")
                                   + ", where things are and where the light comes from", forbid=forbid,
                                   no_style=NO_STYLE, ex_in=EXAMPLE_IN, ex_out=EXAMPLE_NL)
    return _TRANSLATE_SYS


# ---------------- 字典 ----------------
_D = None
_lock = threading.Lock()
_CJK = re.compile(r"[㐀-鿿豈-﫿]")


class TagDict:
    """中文标签补全插件的字典：每行 [标签, 类别, 图数, 中文名, 同义词|…, 别名|…, 备注]，标签用下划线。"""

    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            rows = json.load(f)
        self.cat, self.by_zh, self.by_zh_general, self.by_alias = {}, {}, {}, {}
        for tag, cat, count, zh, syn, alias, *_ in rows:
            self.cat[tag] = int(cat)
            for t in [zh] + (syn.split("|") if syn else []):
                if t and (t not in self.by_zh or count > self.by_zh[t][1]):
                    self.by_zh[t] = (tag, count)
                if t and int(cat) == 0 and (t not in self.by_zh_general or count > self.by_zh_general[t][1]):
                    self.by_zh_general[t] = (tag, count)
            for a in (alias.split("|") if alias else []):
                a = a.lower()
                if a and (a not in self.by_alias or count > self.by_alias[a][1]):
                    self.by_alias[a] = (tag, count)
        self.maxlen = min(10, max((len(k) for k in self.by_zh), default=1))

    def canonical(self, tag):
        """标签（空格或下划线都行）→ 字典里的标准写法；不是真实标签返回 None。"""
        n = tp.norm(tag)
        if n in self.cat:
            return n
        hit = self.by_alias.get(n)
        if hit:
            return hit[0]
        for cut in ("es", "s"):   # 复数写法：puddles → puddle
            if n.endswith(cut) and n[:-len(cut)] in self.cat:
                return n[:-len(cut)]
        return None

    def lookup_zh(self, text, limit=30):
        """一段中文里能查到的标签：从左往右，每个位置取字典里最长的中文词。返回 [(中文, 标签)]。"""
        out, seen = [], set()
        for seg in re.split(r"[,，、。;；\n]+", text or ""):
            seg = seg.strip()
            if not _CJK.search(seg):
                continue
            hit = self.by_zh.get(seg)   # 整段对上：角色名、作品名也算（比如「胡桃」）
            if hit and self.cat.get(hit[0]) == 1:
                hit = None
            if hit:
                if hit[0] not in seen:
                    seen.add(hit[0])
                    out.append((seg, hit[0]))
                continue
            # 切词：在所有「字典里有的词」组合里，选「词长平方和」最大的（偏向长词：「弹吉他」整体胜过「上弹」+「吉他」）
            n = len(seg)
            best = [(0, [])] + [None] * n
            for j in range(1, n + 1):
                cand = (best[j - 1][0], best[j - 1][1])   # 第 j 个字不进任何词
                for L in range(2, min(self.maxlen, j) + 1):
                    w = seg[j - L:j]
                    h = self.by_zh_general.get(w)   # 一段话里的词：只用通用标签，免得「女孩」查成同名角色
                    if h and best[j - L][0] + L * L > cand[0]:
                        cand = (best[j - L][0] + L * L, best[j - L][1] + [(w, h[0])])
                best[j] = cand
            for w, tag in best[n][1]:
                if tag not in seen:
                    seen.add(tag)
                    out.append((w, tag))
            if len(out) >= limit:
                break
        return out[:limit]


def tag_dict():
    """找不到字典插件时返回 None（不查字典，其他照常）。"""
    global _D
    with _lock:
        if _D is None:
            p = os.path.join(locate.sibling_plugin("comfyui-zh-tag-autocomplete"), "data", "tags.json")
            try:
                _D = TagDict(p)
            except (OSError, ValueError, TypeError):
                _D = False
        return _D or None


# ---------------- 组装和收尾 ----------------
_QUALITY_RE = re.compile(r"^(masterpiece|best_quality|good_quality|high_quality|normal_quality|low_quality|worst_quality|"
                         r"amazing_quality|very_aesthetic|aesthetic|score_\d+(_up)?|year_\d{4}|newest|recent|mid|early|old|"
                         r"safe|sensitive|nsfw|explicit|questionable|general|highres|absurdres|lowres)$")


def user_message(prompt, extra="", hints=()):
    u = "User prompt:\n" + (prompt or "").strip()
    if hints:
        u += "\n\nDanbooru tags found in the dictionary for the user's Chinese words (use the ones that match the meaning): " + \
             "; ".join(f"{zh} = {tag.replace('_', ' ')}" for zh, tag in hints)
    if extra and extra.strip():
        u += "\n\nExtra requirements from the user: " + extra.strip()
    return u


def prepare(mode, prompt, extra, on, use_dict=True):
    """返回 (system, user, hints)。hints 是字典查到的 [(中文, 标签)]。"""
    hints = []
    if use_dict and mode in (ANIMA, TRANSLATE):
        d = tag_dict()
        if d:
            hints = d.lookup_zh(prompt)
    return system_prompt(mode, on), user_message(prompt, extra, hints), hints


def finish(text, mode, prompt, on, hints=(), use_dict=True, log=print):
    """清理 AI 的输出。Anima 写法：标签查字典核对、按条件开关去掉不许加的类别（用户自己写的、字典查到的都保留）。"""
    if mode != ANIMA:
        return tp.clean_prompt(text, tags_mode=False)
    s = tp.clean_prompt(text, tags_mode=True)
    m = re.search(r"\.\s+(?=[A-Z])", s)
    head, tail = (s[:m.start()], s[m.end():]) if m else (s, "")
    d = tag_dict() if use_dict else None
    mine = {tp.norm(t) for t in tp.split(prompt)} | {tag for _, tag in hints}
    if d:
        mine |= {c for c in (d.canonical(t) for t in tp.split(prompt)) if c}
    keep, fake, cut = [], [], []
    for t in tp.split(head):
        n = tp.norm(t)
        if n in mine or _QUALITY_RE.match(n):
            keep.append(t)
            continue
        if d:
            c = d.canonical(t)
            if not c:
                fake.append(t)
                continue
            if d.cat.get(c) == 1:   # 画师
                fake.append(t)
                continue
            n = c
            t = tp.escape_parens(c.replace("_", " ")) if "(" in c else c.replace("_", " ")
        blocked = [k for k, _, test in CONDS if not on.get(k, True) and test(n)]
        if blocked:
            cut.append(t)
            continue
        keep.append(t)
    if fake:
        log(f"[WWY反推] 字典里没有、已删掉：{', '.join(fake)}")
    if cut:
        log(f"[WWY反推] 按开关去掉：{', '.join(cut)}")
    seen, out = set(), []
    for t in keep:
        n = tp.norm(t)
        if n not in seen:
            seen.add(n)
            out.append(t)
    head = ", ".join(out)
    return (head + ". " + tail.strip()) if tail.strip() else head
