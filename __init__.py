# -*- coding: utf-8 -*-
"""
WWY 反推：照 WWY 启动器「图片反推」页的逻辑做的 ComfyUI 节点。

  反推：WD14 标签      wd-eva02-large / wd-vit / PixAI……，模型和启动器、LoRA 训练器共用
  反推：自然语言        ToriiGate 0.5 / JoyCaption / Qwen3-VL 8B·4B，或启动器设置的 AI 接口
  反推：筛选标签        去掉画师 / 角色 / 作品 / 元标签 / 人物固有特征，黑名单、白名单
  合并提示词            主提示词 + 反推标签 + 描述，去重
  AI 优化提示词         本地 Qwen3-VL 或 AI 接口，把提示词扩写成 Anima / 自然语言的写法
"""
import os
import tempfile

import numpy as np
from PIL import Image

from .wwy import textproc as tp
from .wwy import enhance as en
from .wwy import vlm_core as vc

CATEGORY = "WWY 反推"

VLM_CHOICES = {
    "ToriiGate 0.5（二次元专精）": "toriigate",
    "JoyCaption Beta One（通用，描述细致）": "joycaption",
    "Qwen3-VL-8B（无审查）": "qwen3vl8b",
    "Qwen3-VL-4B（显存小）": "qwen3vl4b",
    "API（启动器或 config.json 里的设置）": "api",
}
TEXT_CHOICES = {
    "Qwen3-VL-8B（本地）": "qwen3vl8b",
    "Qwen3-VL-4B（本地，显存小）": "qwen3vl4b",
    "API（启动器或 config.json 里的设置）": "api",
}
PRESETS = {label: key for key, label in vc.PROMPT_LABELS}
QUANTS = {"自动": "auto", "原精度 bf16": "bf16", "4bit 量化（显存小）": "4bit"}


def _pil(image):
    """IMAGE 张量 [B,H,W,C] 0~1 → 第一张 PIL"""
    a = (image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _err(e):
    return RuntimeError(f"[WWY反推] {e}")


class WWYTagger:
    @classmethod
    def INPUT_TYPES(cls):
        from .wwy import wd14
        return {"required": {
            "图像": ("IMAGE",),
            "模型": (wd14.choices(), {"default": wd14.DEFAULT}),
            "通用阈值": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            "角色阈值": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
            "转义括号": ("BOOLEAN", {"default": True, "tooltip": "saber (fate) → saber \\(fate\\)，直接当提示词用时要转义"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("标签",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "WD14 / PixAI 标签反推（只看批次里的第一张图）。模型没下载时自动下载，和 WWY 启动器、LoRA 训练器共用。"

    def run(self, 图像, 模型, 通用阈值, 角色阈值, 转义括号):
        from .wwy import wd14
        try:
            r = wd14.get(模型).tag_image(_pil(图像), 通用阈值, 角色阈值)
        except Exception as e:
            raise _err(e)
        tags = r["tags"]
        if 转义括号:
            tags = [tp.escape_parens(t) for t in tags]
        s = ", ".join(tags)
        print(f"[WWY反推] WD14：{s}", flush=True)
        return (s,)


class WWYCaption:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "图像": ("IMAGE",),
            "模型": (list(VLM_CHOICES), {"default": "ToriiGate 0.5（二次元专精）"}),
            "指令": (list(PRESETS), {"default": "详细描述（通用）",
                                   "tooltip": "ToriiGate 用自己的指令格式（下面的长短、识别角色名），其他模型和 API 用这里选的"}),
            "自定义指令": ("STRING", {"multiline": True, "default": "", "tooltip": "「指令」选自定义时用这里的文字"}),
            "ToriiGate长短": (["长", "短"], {"default": "短"}),
            "不描述画风": ("BOOLEAN", {"default": True, "tooltip": "反推结果要并进提示词时打开：不写画风、画法、画师"}),
            "识别角色名": ("BOOLEAN", {"default": False, "tooltip": "ToriiGate 试着认出角色、写出名字；认不准时会编一个名字，默认关"}),
            "最多字数": ("INT", {"default": 400, "min": 32, "max": 2048, "step": 16, "tooltip": "max_new_tokens"}),
            "温度": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05}),
            "精度": (list(QUANTS), {"default": "自动"}),
            "用完释放显存": ("BOOLEAN", {"default": True, "tooltip": "反推完结束模型进程，后面采样时显存够用"}),
        }, "optional": {
            "参考标签": ("STRING", {"forceInput": True, "tooltip": "接 WD14 的标签：交给模型参考（启动器的「标签 + 自然语言」）"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("描述",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "看图写一段英文描述（只看第一张图）。本地模型跑在单独的进程里，LoRA 训练器的环境优先。"

    def run(self, 图像, 模型, 指令, 自定义指令, ToriiGate长短, 不描述画风, 识别角色名, 最多字数, 温度, 精度, 用完释放显存,
            参考标签=""):
        key = VLM_CHOICES[模型]
        opts = {"preset": PRESETS[指令], "prompt": 自定义指令, "torii_format": "long" if ToriiGate长短 == "长" else "short",
                "use_names": 识别角色名}
        tags = tp.split(参考标签) if 参考标签 else None
        pil = _pil(图像)
        try:
            if key == "api":
                from .wwy import api
                user = vc.user_prompt(opts) + (" " + tp.NO_STYLE_CAPTION if 不描述画风 else "")
                if tags:
                    user += "\nReference tags (may contain mistakes): " + ", ".join(tags)
                text = api.chat(vc.DEFAULT_SYSTEM, user, image=pil, max_tokens=最多字数, temperature=温度)
            else:
                from .wwy.vlm_service import SERVICE
                system, user = vc.build_prompt(key, opts, tags)
                if 不描述画风:   # ToriiGate 的指令是「# 小标题」分段的格式，照着加一段
                    user = user.rstrip() + ("\n\n# Style:\n" if key == "toriigate" else " ") + tp.NO_STYLE_CAPTION
                fd, path = tempfile.mkstemp(suffix=".png", prefix="wwy_cap_")
                os.close(fd)
                try:
                    pil.save(path)
                    text = SERVICE.run(key, QUANTS[精度], [{"file": path, "system": system, "user": user}],
                                       max_tokens=最多字数, temperature=温度, unload=用完释放显存)[0]
                finally:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
        except vc.CaptionError as e:
            raise _err(e)
        text = vc.clean_text(text)
        print(f"[WWY反推] 描述：{text}", flush=True)
        return (text,)


class WWYTagFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "标签": ("STRING", {"forceInput": True}),
            "去掉画师": ("BOOLEAN", {"default": True}),
            "去掉评分和元标签": ("BOOLEAN", {"default": True, "tooltip": "general / sensitive、highres、commentary 之类"}),
            "去掉角色名": ("BOOLEAN", {"default": False}),
            "去掉作品名": ("BOOLEAN", {"default": False}),
            "去掉人物特征": ("BOOLEAN", {"default": False, "tooltip": "发色发型、瞳色、兽耳尾巴角、体型肤色：只想借参考图的动作构图、换成自己的角色时打开"}),
            "黑名单": ("STRING", {"multiline": True, "default": "", "tooltip": "要去掉的标签，逗号或换行分隔，支持 * 通配，例如 *background*"}),
            "只保留": ("STRING", {"multiline": True, "default": "", "tooltip": "不为空时只留下匹配的标签，支持 * 通配"}),
            "最多几个": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "0 = 不限（按反推的可信度从高到低留）"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("标签",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "按类别筛掉反推出来的标签。类别查的是中文标签字典（comfyui-zh-tag-autocomplete）的 tags.json。"

    def run(self, 标签, 去掉画师, 去掉评分和元标签, 去掉角色名, 去掉作品名, 去掉人物特征, 黑名单, 只保留, 最多几个):
        s = tp.filter_tags(标签, 去掉画师, 去掉角色名, 去掉作品名, 去掉评分和元标签, 去掉人物特征, 黑名单, 只保留, 最多几个)
        print(f"[WWY反推] 筛选后：{s}", flush=True)
        return (s,)


class WWYPromptMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "去重": ("BOOLEAN", {"default": True}),
            "标签放前面": ("BOOLEAN", {"default": False, "tooltip": "关：主提示词在前（质量词在最前面）；开：反推标签在前"}),
        }, "optional": {
            "主提示词": ("STRING", {"forceInput": True}),
            "标签": ("STRING", {"forceInput": True}),
            "描述": ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "主提示词 + 反推标签 + 描述句子。没接的（或所在分组被绕过的）直接跳过。"

    def run(self, 去重, 标签放前面, 主提示词="", 标签="", 描述=""):
        if 描述 and 标签 and 描述.strip() == 标签.strip():
            描述 = ""   # 「反推：自然语言」被绕过时，会把它的「参考标签」原样传过来
        s = tp.merge(主提示词 or "", 标签 or "", 描述 or "", 去重, 标签放前面)
        print(f"[WWY反推] 合并后：{s}", flush=True)
        return (s,)


class WWYPromptEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "提示词": ("STRING", {"forceInput": True}),
            "写法": (en.MODES, {"default": en.ANIMA}),
            "模型": (list(TEXT_CHOICES), {"default": "Qwen3-VL-8B（本地）"}),
            "补外貌": ("BOOLEAN", {"default": True, "tooltip": "关掉：AI 不新加发型、发色、瞳色这类特征（用了角色 LoRA 时常关）。你自己写的照样保留"}),
            "补服装": ("BOOLEAN", {"default": True, "tooltip": "关掉：AI 不新加衣服、鞋子、饰品。你自己写的照样保留"}),
            "补动作表情": ("BOOLEAN", {"default": True, "tooltip": "关掉：AI 不新加动作、姿势、表情。你自己写的照样保留"}),
            "补背景": ("BOOLEAN", {"default": True, "tooltip": "关掉：AI 不新加场景、背景。你自己写的照样保留"}),
            "补光线构图": ("BOOLEAN", {"default": True, "tooltip": "关掉：AI 不新加光线、景别、视角。你自己写的照样保留"}),
            "查字典": ("BOOLEAN", {"default": True, "tooltip": "用中文标签补全插件的字典：写之前把你写的中文查成标准标签交给 AI，"
                                                          "写完逐个核对 AI 给的标签，字典里没有的（编出来的）和画师标签删掉。只对 Anima 写法核对"}),
            "额外要求": ("STRING", {"multiline": True, "default": "", "tooltip": "比如：画面更有故事感、换成夜晚"}),
            "最多字数": ("INT", {"default": 400, "min": 32, "max": 2048, "step": 16}),
            "温度": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.5, "step": 0.05}),
            "种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "tooltip": "换一个数字就重新写一遍"}),
            "用完释放显存": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    DESCRIPTION = "让大模型把提示词扩写成更适合出图的写法（不会加画师和画风）。可以关掉不想让它补的类别；Anima 写法会查字典核对标签。"

    def run(self, 提示词, 写法, 模型, 额外要求, 最多字数, 温度, 种子, 用完释放显存,
            补外貌=True, 补服装=True, 补动作表情=True, 补背景=True, 补光线构图=True, 查字典=True):
        key = TEXT_CHOICES[模型]
        on = {"补外貌": 补外貌, "补服装": 补服装, "补动作表情": 补动作表情, "补背景": 补背景, "补光线构图": 补光线构图}
        system, user, hints = en.prepare(写法, 提示词, 额外要求, on, 查字典)
        if hints:
            print("[WWY反推] 字典查到：" + "；".join(f"{zh} → {tag}" for zh, tag in hints), flush=True)
        if 种子:
            user += f"\n\n(variation #{种子})"
        try:
            if key == "api":
                from .wwy import api
                text = api.chat(system, user, max_tokens=最多字数, temperature=温度)
            else:
                from .wwy.vlm_service import SERVICE
                text = SERVICE.run(key, "auto", [{"file": "", "id": "t0", "system": system, "user": user}],
                                   max_tokens=最多字数, temperature=温度, unload=用完释放显存)[0]
        except vc.CaptionError as e:
            raise _err(e)
        text = en.finish(text, 写法, 提示词, on, hints, 查字典, log=lambda m: print(m, flush=True))
        print(f"[WWY反推] AI 优化后：{text}", flush=True)
        return (text,)


NODE_CLASS_MAPPINGS = {
    "WWYTagger": WWYTagger,
    "WWYCaption": WWYCaption,
    "WWYTagFilter": WWYTagFilter,
    "WWYPromptMerge": WWYPromptMerge,
    "WWYPromptEnhance": WWYPromptEnhance,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WWYTagger": "反推：WD14 标签",
    "WWYCaption": "反推：自然语言",
    "WWYTagFilter": "反推：筛选标签",
    "WWYPromptMerge": "合并提示词",
    "WWYPromptEnhance": "AI 优化提示词",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
