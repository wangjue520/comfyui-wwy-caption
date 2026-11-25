# -*- coding: utf-8 -*-
"""WWY LoRA 堆：想挂几个挂几个，每个 LoRA 一个文件、一个强度、一个开关。

界面上的控件全是 ComfyUI 自带的下拉框 / 数字框 / 开关（web/lora_stack.js 按需加减），
不像 rgthree 的 Power Lora Loader 那样自己画控件——那个在新前端下控件宽度会被改掉，
点箭头、点数字、点开关时灵时不灵（rgthree-comfy issue #731）。

每个 LoRA 的控件名是 lora_1 / strength_1 / on_1、lora_2 / …，个数不固定，
所以可选输入用「什么名字都收」的字典（和 rgthree 的做法一样）。
"""
import re

import comfy.sd
import comfy.utils
import folder_paths


class AnyType(str):
    """和任何类型都算匹配"""

    def __ne__(self, other):
        return False


ANY = AnyType("*")


class AnyKeys(dict):
    """可选输入：data 里的照常，别的名字（lora_N、strength_N、on_N）也都收下"""

    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        if key in self.data:
            return self.data[key]
        return (ANY,)


ROW = re.compile(r"^(lora|strength|on)_(\d+)$")


def rows_of(kwargs):
    """{lora_1: …, strength_1: …, on_1: …} → [(文件名, 强度, 开没开), …]，按编号排"""
    rows = {}
    for k, v in kwargs.items():
        m = ROW.match(k)
        if m:
            rows.setdefault(int(m.group(2)), {})[m.group(1)] = v
    out = []
    for i in sorted(rows):
        r = rows[i]
        out.append((r.get("lora") or "None", float(r.get("strength", 1.0)), bool(r.get("on", True))))
    return out


class WWYLoraStack:
    DESCRIPTION = ("LoRA 堆：点「➕ 加一个 LoRA」想挂几个挂几个；每个 LoRA 选文件、调强度，开关关掉 = 先不用（文件和强度留着）。"
                   "右键节点可以删掉中间某一个。只接 model 就只改模型；接了 clip，带文本编码器部分的 LoRA 也会改它。")

    def __init__(self):
        self.cache = {}   # 路径 → 读进来的 LoRA，上一次用到的留着，不用每次重读

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"model": ("MODEL",)},
            "optional": AnyKeys({"clip": ("CLIP",)}),
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = "WWY"

    def apply(self, model, clip=None, **kwargs):
        used = {}
        for name, strength, on in rows_of(kwargs):
            if not on or name == "None" or strength == 0:
                continue
            path = folder_paths.get_full_path("loras", name)
            if path is None:
                raise RuntimeError(f"[WWY LoRA 堆] 找不到 LoRA：{name}（不在 loras 文件夹里，换一个或者把这一行删掉）")
            lora = self.cache.get(path)
            if lora is None:
                lora = comfy.utils.load_torch_file(path, safe_load=True)
            used[path] = lora
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, strength, strength if clip is not None else 0)
            print(f"[WWY LoRA 堆] {name} × {strength:g}", flush=True)
        self.cache = used
        return (model, clip)
