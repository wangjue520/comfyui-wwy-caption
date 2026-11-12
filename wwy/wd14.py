# -*- coding: utf-8 -*-
"""WD14 / PixAI 标签反推：推理照搬启动器 wd14_tagger_core.py，模型和启动器、训练器共用。"""
import os
import threading

from . import locate
from . import wd14_core as wt

# 启动器的三个 + 训练器里常见的两个
CATALOG = dict(wt.MODEL_CATALOG)
CATALOG.setdefault("wd-swinv2-tagger-v3", {"label": "wd-swinv2-tagger-v3（速度快）", "repo": "SmilingWolf/wd-swinv2-tagger-v3"})
CATALOG.setdefault("wd-vit-large-tagger-v3", {"label": "wd-vit-large-tagger-v3（精度高）", "repo": "SmilingWolf/wd-vit-large-tagger-v3"})
DEFAULT = "wd-eva02-large-tagger-v3"

_cache = {}
_lock = threading.Lock()


def _sibling_pair(key):
    """pythongosssss 的 comfyui-wd14-tagger 插件把模型放成 models/<名字>.onnx + <名字>.csv，也认一下。"""
    base = os.path.join(locate.sibling_plugin("comfyui-wd14-tagger"), "models", key)
    if os.path.isfile(base + ".onnx") and os.path.isfile(base + ".csv"):
        return base + ".onnx", base + ".csv"
    return None


def find(key):
    """(onnx, csv) 或 None"""
    info = CATALOG[key]
    for d in locate.wd14_dirs(key, info["repo"]):
        o, c = os.path.join(d, "model.onnx"), os.path.join(d, "selected_tags.csv")
        if os.path.isfile(o) and os.path.isfile(c):
            return o, c
    return _sibling_pair(key)


def choices():
    return list(CATALOG.keys())


class _Tagger(wt.WD14Tagger):
    """WD14Tagger 只认「目录里的 model.onnx + selected_tags.csv」，这里改成直接给两个文件。"""

    def __init__(self, onnx, csv_path):
        import onnxruntime as ort
        d = os.path.dirname(csv_path)
        self.session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        outputs = [o.name for o in self.session.get_outputs()]
        self.output_name = "prediction" if "prediction" in outputs else outputs[0]
        shape = self.session.get_inputs()[0].shape
        if os.path.basename(csv_path) == "selected_tags.csv":
            self.tag_names, self.rating_idx, self.general_idx, self.character_idx = wt.load_labels(d)
            self.ips = wt.load_ips(d)
        else:
            self.tag_names, self.rating_idx, self.general_idx, self.character_idx = _load_labels_file(csv_path)
            self.ips = {}
        if len(shape) == 4 and shape[1] == 3:
            self.nchw = True
        elif len(shape) == 4 and shape[3] == 3:
            self.nchw = False
        else:
            self.nchw = bool(self.ips)
        spatial = shape[2] if self.nchw else (shape[1] if len(shape) == 4 else None)
        self.target_size = spatial if isinstance(spatial, int) and spatial > 1 else 448


def _load_labels_file(path):
    import csv
    names, r, g, c = [], [], [], []
    with open(path, encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            names.append(row["name"])
            cat = int(row["category"])
            (r if cat == 9 else c if cat == 4 else g).append(i)
    return names, r, g, c


def get(key, log=print):
    with _lock:
        if key in _cache:
            return _cache[key]
        pair = find(key)
        if not pair:
            d = os.path.join(locate.comfy_wd14_dir(), key)
            log(f"[WWY反推] 本地没有 {key}，下载到 {d}")
            _download(key, d, log)
            pair = (os.path.join(d, "model.onnx"), os.path.join(d, "selected_tags.csv"))
        t = _Tagger(*pair)
        _cache[key] = t
        return t


def _download(key, d, log):
    """wd14_core.download_model 下到 <base>/wd_tagger_models/<key>，这里要下到 ComfyUI/models/wd14_tagger/<key>。"""
    orig = wt.model_cache_dir
    try:
        wt.model_cache_dir = lambda _base, _key: d
        if key not in wt.MODEL_CATALOG:
            wt.MODEL_CATALOG[key] = CATALOG[key]
        wt.download_model("", key, log_cb=lambda t: log("[WWY反推] " + t))
    finally:
        wt.model_cache_dir = orig
