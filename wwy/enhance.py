# -*- coding: utf-8 -*-
"""
AI 优化提示词：写法、条件开关、查字典。

写法参考 comfyui-mcp 的 anima-base / krea2-txt2img 两个 skill：
  · Anima：质量词 → 人数和主体标签 → 内容标签 → 2 到 4 句英文描述；提示词太短太空时容易画出不想要的东西。
  · Krea 2：用视觉语言模型（Qwen3-VL）读提示词，写成完整、具体的英文段落，像在描述一张已经存在的图。
条件开关：关掉的那一类（外貌、服装、动作表情、背景、光线构图）AI 不许新加；用户自己写的照样保留。
  标签里的按类别删，句子里的按「哪些词是用户写过的」删掉 AI 补的短语，删不干净就删整句。
画师串和画风词：用户自己写的不交给 AI，原样拿出来，写完再原样放回（Anima 放在质量词后面，其他写法放最前面）。
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
            "'realistic') on your own; the user's own style words stay as the user wrote them. Never add tags or sentences "
            "the user did not ask for that change the rating.")

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
    """返回 (system, user, ctx)。ctx 给 finish 用：字典查到的 hints、原样保留的画师串和画风词。"""
    keep, rest = split_protected(prompt)
    d = tag_dict() if use_dict else None
    found = d.lookup_zh(rest) if d else []
    hints = found if mode in (ANIMA, TRANSLATE) else []
    ctx = {"hints": hints, "found": found, "protected": keep, "rest": rest, "extra": extra or "",
           "cjk_style": bool(_ZH_STYLE.search(rest))}
    return system_prompt(mode, on), user_message(rest, extra, hints), ctx


def finish(text, mode, prompt, on, ctx=None, use_dict=True, log=print):
    """清理 AI 的输出。
    Anima 写法：标签查字典核对、按条件开关去掉不许加的类别（用户自己写的、字典查到的都保留）；
    Anima 和纯自然语言：句子里按开关删掉 AI 补的内容；最后把用户的画师串和画风词原样放回去。"""
    ctx = ctx or {}
    hints = ctx.get("hints") or []
    protected = ctx.get("protected") or []
    rest = ctx.get("rest", prompt)
    # 用户写了中文画风词（会被 AI 翻成英文）：这次不按画风词清理，免得把用户自己的删掉
    style_ok = ctx.get("cjk_style", False)
    if mode != ANIMA:
        s = tp.clean_prompt(text, tags_mode=False)
        if mode == NATURAL:
            s = scrub_sentences(s, on, _user_words(rest, ctx), log)
        return _put_back(s, protected, mode)
    s = tp.clean_prompt(text, tags_mode=True, drop_styles=not style_ok)
    m = re.search(r"\.\s+(?=[A-Z])", s)
    head, tail = (s[:m.start()], s[m.end():]) if m else (s, "")
    d = tag_dict() if use_dict else None
    mine = {tp.norm(t) for t in tp.split(rest)} | {tag for _, tag in hints}
    if d:
        mine |= {c for c in (d.canonical(t) for t in tp.split(rest)) if c}
    prot = {tp.norm(t) for t in protected}
    keep, fake, cut = [], [], []
    for t in tp.split(head):
        n = tp.norm(t)
        if n in prot:   # AI 自己又写了一遍：放回去的时候用用户的原样
            continue
        if n in mine or _QUALITY_RE.match(n):
            keep.append(t)
            continue
        if style_ok and (tp._STYLE_RE.search(t) or _MEDIUM_RE.search(t)):
            keep.append(t)   # 用户中文画风词翻过来的，不按字典删
            continue
        if d:
            c = d.canonical(t)
            if not c:
                fake.append(t)
                continue
            if d.cat.get(c) == 1:   # 画师（用户自己的已经拿出来了，这里的是 AI 加的）
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
    if tail.strip():
        tail = scrub_sentences(tail.strip(), on, _user_words(rest, ctx), log)
    head = ", ".join(_insert_after_quality(out, protected))
    return (head + ". " + tail.strip()) if tail.strip() else head


# ---------------- 画师串、画风词：原样保留 ----------------
_ZH_STYLE_WORDS = ("水彩|油画|素描|速写|厚涂|平涂|赛璐璐|半写实|写实|真人|照片|摄影|电影感|电影光|电影质感|像素|水墨|国风|古风|线稿|草图|"
                   "三渲二|3D|3d|卡通|动漫|二次元|胶片|插画|板绘|手绘|低多边形|赛博朋克|蒸汽波|浮世绘|版画|黏土|粘土|彩铅|马克笔|"
                   "印象派|极简|扁平|美式|日系|韩系|欧美|复古|油彩|丙烯|水粉|蜡笔|铅笔")
_ZH_STYLE = re.compile(r"(?:" + _ZH_STYLE_WORDS + r")(?:风格|画风|风|质感|效果|渲染)?|画风|风格")
_ZH_STYLE_ONLY = re.compile(r"^(?:" + _ZH_STYLE_WORDS + r"|风格|画风|风|质感|效果|渲染|的|感|式|\s)+$")
_ARTIST_PREFIX = re.compile(r"^\(?\s*(?:@|by\s|artist\s*[:：])", re.I)
_MEDIUM_RE = re.compile(r"\(medium\)|\\\(medium\\\)|^(?:monochrome|greyscale|grayscale|lineart|flat color|cel shading|anime coloring|"
                        r"pixel art|traditional media|ukiyo-e|impressionism|chibi style)$", re.I)


def _is_protected(seg, d):
    t = seg.strip()
    if not t:
        return False
    n = tp.norm(t)
    if _QUALITY_RE.match(n):
        return False   # 质量词照常交给 AI（指令里要它原样放最前面）
    if _ARTIST_PREFIX.search(t):
        return True
    if _CJK.search(t):
        # 只有整段都是画风说法才原样保留（「水彩风格」「新海诚画风」）；混着内容的交给 AI 翻译
        return bool(_ZH_STYLE_ONLY.match(t)) or (len(t) <= 12 and re.search(r"(?:画风|风格)$", t) is not None)
    if len(t.split()) > 6 or re.search(r"[.!?]$", t):
        return False   # 一句话：交给 AI
    cats = d.cat if d else tp.tag_categories()
    c = (d.canonical(t) if d else None) or n
    if cats.get(c) == 1:
        return True
    return bool(tp._STYLE_RE.search(t) or _MEDIUM_RE.search(t))


def split_protected(prompt):
    """用户的提示词 → (原样保留的画师串和画风词, 剩下交给 AI 的部分)。按逗号、换行切，保留原来的写法和权重括号。"""
    d = tag_dict()
    keep, rest = [], []
    for seg in re.split(r"[,，\n]", prompt or ""):
        if not seg.strip():
            continue
        (keep if _is_protected(seg, d) else rest).append(seg.strip())
    return keep, ", ".join(rest)


def _insert_after_quality(tags, protected):
    i = 0
    while i < len(tags) and _QUALITY_RE.match(tp.norm(tags[i])):
        i += 1
    return tags[:i] + list(protected) + tags[i:]


def _put_back(s, protected, mode):
    if not protected:
        return s
    if mode == NATURAL:
        return ", ".join(protected) + ". " + s if s else ", ".join(protected)
    return ", ".join(protected) + (", " + s if s else "")


# ---------------- 句子：关掉的类别，AI 补的删掉 ----------------
def _kw(words):
    return [w.replace(" ", r"\s+") for w in words]


_APPEAR_BODY = ["hair", "eyes", "eye", "pupils", "skin", "complexion", "bangs", "fringe", "eyebrows", "eyelashes", "lips",
                "ears"]
_APPEAR_TRAIT = ["ponytail", "twintails", "twin tails", "pigtails", "braid", "braids", "bun", "ahoge", "sidelocks",
                 "animal ears", "cat ears", "fox ears", "dog ears", "rabbit ears", "bunny ears", "tail", "horns",
                 "horn", "wings", "halo", "freckles", "mole", "fang", "fangs", "heterochromia", "breasts"]
_CLOTHES = ["dress", "skirt", "shirt", "t-shirt", "blouse", "uniform", "jacket", "coat", "hoodie", "sweater", "cardigan",
            "vest", "kimono", "yukata", "hakama", "apron", "bikini", "swimsuit", "leotard", "bodysuit", "pants", "shorts",
            "jeans", "trousers", "thighhighs", "thigh-highs", "stockings", "pantyhose", "socks", "legwear", "shoes", "boots",
            "sandals", "heels", "high heels", "loafers", "sneakers", "gloves", "hat", "cap", "beret", "hood", "necktie",
            "bow tie", "bowtie", "neckerchief", "scarf", "choker", "collar", "ribbon", "ribbons", "bow", "hair ornament",
            "hairclip", "hair clip", "hairband", "headband", "earrings", "jewelry", "necklace", "bracelet", "armor",
            "cape", "cloak", "robe", "bra", "panties", "underwear", "lingerie", "serafuku", "suit", "tuxedo", "camisole",
            "tank top", "crop top", "sleeves", "sleeve", "frills", "lace", "garter", "belt", "glasses", "veil", "tiara",
            "crown", "clothes", "clothing", "outfit", "costume", "attire", "garment", "wears", "wearing", "dressed",
            "nude", "naked", "topless", "barefoot"]
_POSE = ["smile", "smiles", "smiling", "smiled", "grin", "grins", "grinning", "laugh", "laughs", "laughing", "blush",
         "blushes", "blushing", "tears", "crying", "cries", "frown", "frowns", "frowning", "pout", "pouting", "wink",
         "winks", "winking", "expression", "gaze", "looking at the viewer", "looks at the viewer", "pose", "poses",
         "posing", "peace sign", "crossed arms", "arms crossed", "hand on her hip", "hands on her hips", "head tilted",
         "tilts her head", "tilting her head"]
_SCENE = ["background", "backdrop", "setting", "scenery", "landscape", "indoors", "outdoors", "room", "street", "city",
          "cityscape", "forest", "woods", "beach", "ocean", "sea", "lake", "river", "school", "classroom", "library",
          "bedroom", "cafe", "café", "station", "park", "garden", "field", "meadow", "mountain", "mountains", "sky",
          "clouds", "cloud", "night", "sunset", "sunrise", "dusk", "dawn", "rain", "snow", "building", "buildings",
          "window", "windows", "tree", "trees", "grass", "flowers", "stage", "bookshelf", "bookshelves", "shelves",
          "table", "desk", "chair", "bench", "bed", "sofa", "couch", "road", "bridge", "shrine", "temple", "alley",
          "rooftop", "stars", "water", "wall", "walls", "floor", "corridor", "hallway"]
_LIGHT = ["lighting", "lit", "backlit", "backlighting", "sunlight", "moonlight", "candlelight", "lamplight", "daylight",
          "glow", "glows", "glowing", "shadow", "shadows", "lens flare", "light rays", "sunbeam", "sunbeams", "bokeh",
          "depth of field", "close-up", "closeup", "shot", "angle", "camera", "framing", "framed", "frame",
          "composition", "viewpoint", "perspective", "portrait", "upper body", "full body", "cowboy shot",
          "from above", "from below", "from behind", "low angle", "high angle", "light falls", "light streams",
          "light pours", "light shines", "warm light", "soft light", "cold light"]
_SENT_CATS = {   # 开关名: (关键词, 判定方式)
    "补外貌": (_APPEAR_BODY + _APPEAR_TRAIT, "appear"),
    "补服装": (_CLOTHES, "attr"),
    "补动作表情": (_POSE, "word"),
    "补背景": (_SCENE, "word"),
    "补光线构图": (_LIGHT, "word"),
}
_COMPOUND = re.compile(r"\b([a-z]+)-(haired|eyed|skinned|eared|tailed|horned)\b", re.I)
# 修饰词里算「具体属性」的：颜色、长短、发型……AI 自己补上这些就算新加了外貌 / 服装细节
_ATTR = set("""black white blonde blond golden gold brown red blue green purple pink silver grey gray orange yellow aqua
cyan teal violet crimson scarlet azure amber emerald platinum ash dark pale light multicolored two-tone gradient long short
medium very twin messy wavy curly straight spiky fluffy large small big huge flat tall petite slender curvy tanned fair
sleeveless frilly frilled pleated striped plaid lace see-through tight loose""".split())
_STOP = set("""a an the her his their its my your our with and or has have had is are was were be being been wearing wears
wear worn in on at of to by she he they it from into over under through behind beside near that which while as both very
slightly softly gently also neatly""".split())
_SYN = {"blond": "blonde", "golden": "blonde", "gold": "blonde", "gray": "grey", "eye": "eyes"}
_PREP_LEFT = re.compile(r"(?:,\s*)?\b(?:with|wearing|in|at|on|inside|under|beneath|by|near|beside|behind|through|against|"
                        r"under|across|before|among|amid|dressed in|clad in)\s+(?:(?:a|an|the|her|his|their|its)\s+)?$", re.I)
_AND_LEFT = re.compile(r"(?:\s*,\s*(?:and\s+)?|\s+and\s+)(?:(?:a|an|the|her|his|their|its)\s+)?$", re.I)
_AND_RIGHT = re.compile(r"^(?:\s*,)?\s+and\s+(?:(?:a|an|the|her|his|their|its)\s+)?", re.I)


def _w(x):
    x = x.lower()
    x = _SYN.get(x, x)
    return x


def _forms(x):
    x = _w(x)
    out = {x}
    for cut in ("es", "s"):
        if x.endswith(cut) and len(x) > len(cut) + 2:
            out.add(x[:-len(cut)])
    return out


def _user_words(rest, ctx):
    """用户写过的英文词（包括字典把中文查成的标签）：句子里这些词组成的说法算用户自己的。"""
    text = " ".join([rest, ctx.get("extra", "")] + [t for _, t in (ctx.get("found") or [])] +
                    [t for _, t in (ctx.get("hints") or [])])
    words = set()
    for x in re.findall(r"[A-Za-z][A-Za-z'-]*", text.replace("_", " ")):
        for f in _forms(x):
            words.add(f)
        for part in x.split("-"):
            words |= _forms(part)
    return words


def _spans(sent, key):
    """一句话里这一类的说法：[(开始, 结束, 关键词, 前面的修饰词)]。"""
    words, how = _SENT_CATS[key]
    pat = re.compile(r"\b(?:" + "|".join(sorted(_kw(words), key=len, reverse=True)) + r")\b", re.I)
    out = []
    for m in pat.finditer(sent):
        mods, start = [], m.start()
        for w in reversed(list(re.finditer(r"[A-Za-z][A-Za-z'-]*", sent[:m.start()]))):
            between = sent[w.end():start]
            if between.strip() or len(mods) >= 3 or w.group().lower() in _STOP:
                break
            mods.insert(0, w.group())
            start = w.start()
        out.append((start, m.end(), m.group(), mods, how))
    if key == "补外貌":
        for m in _COMPOUND.finditer(sent):
            out.append((m.start(), m.end(), m.group(2), [m.group(1)], "compound"))
    return out


def _allowed(kw, mods, how, uw):
    kws = re.split(r"\s+", kw)
    have_kw = all(_forms(k) & uw for k in kws)
    new_attr = [x for x in mods if _w(x) in _ATTR and not (_forms(x) & uw)]
    if how == "compound":   # silver-haired：颜色是用户写过的才算
        return not new_attr
    if how == "appear":
        if " ".join(kws).lower() in _APPEAR_TRAIT:
            return have_kw and not new_attr
        return not new_attr   # hair / eyes 本身不算外貌，带了用户没写过的颜色长短才算
    if how == "attr":
        return have_kw and not new_attr
    return have_kw


def _remove(sent, a, b):
    """删掉 sent[a:b] 这个说法，连带它前面的 with / in / and 之类，尽量让句子还通顺。返回新句子或 None（删不干净）。"""
    left, right = sent[:a], sent[b:]
    m = _AND_LEFT.search(left)
    if m and m.start() > 0 and not _PREP_LEFT.search(left):
        return left[:m.start()] + right
    m = _PREP_LEFT.search(left)
    if m:
        r = _AND_RIGHT.match(right)
        if r:   # with long hair and blue eyes → 只删 long hair and：with blue eyes
            return left + right[r.end():]
        return left[:m.start()] + right
    return None


def _tidy(s):
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s+([,.;!?])", r"\1", s)
    s = re.sub(r",\s*([,.;!?])", r"\1", s)
    s = re.sub(r"\b(an?)\s+(?=[,.;!?]|$)", "", s)
    s = re.sub(r"\ban (?=[b-df-hj-np-tv-z])", "a ", s, flags=re.I)
    s = re.sub(r"\ba (?=[aeiou])", "an ", s, flags=re.I)
    s = s.strip(" ,;")
    return s[:1].upper() + s[1:] if s else s


def scrub_sentences(text, on, uw, log=print):
    """关掉的类别：句子里 AI 新补的说法删掉（用户写过的词组成的说法保留）。先删短语，删不干净就删整句（至少留一句）。"""
    off = [k for k in _SENT_CATS if not on.get(k, True)]
    if not off or not text.strip():
        return text
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    out, removed = [], []
    for sent in sents:
        s, drop, part = sent, False, []
        for _ in range(12):   # 一次删一个，删完重新找
            bad = [sp for k in off for sp in _spans(s, k) if not _allowed(sp[2], sp[3], sp[4], uw)]
            if not bad:
                break
            a, b, _, _, how = max(bad, key=lambda x: x[0])   # 从后往前删，前面的位置不变
            n = s[:a] + s[b:] if how == "compound" else _remove(s, a, b)   # silver-haired girl → girl
            if n is None:
                drop = True
                break
            part.append(s[a:b].strip())
            s = n
        s = _tidy(s)
        if drop or len(s.split()) < 3:
            removed.append(sent.strip())
            out.append(None)
        else:
            removed += part
            out.append(s if s.endswith((".", "!", "?")) else s + ".")
    kept = [x for x in out if x]
    if not kept:   # 全删光了：留第一句删过短语的样子（或原句），别让描述整个没了
        kept = [_tidy(sents[0])]
    if removed:
        log(f"[WWY反推] 按开关从句子里删掉：{' / '.join(removed)}")
    return " ".join(kept)
