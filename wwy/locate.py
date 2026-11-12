# -*- coding: utf-8 -*-
"""
东西放在哪：WWY 启动器、LoRA 训练器、反推模型、跑自然语言模型的 Python 环境。

和启动器「图片反推」页一样的找法：
  · 自然语言模型：训练器的 models/taggers → 启动器的 vlm_models → config.json 里加的目录 → ComfyUI/models/vlm_caption
  · WD14 模型：启动器的 wd_tagger_models → 训练器的 models/taggers → ComfyUI/models/wd14_tagger
  · 跑自然语言模型的环境：训练器的 env（transformers 新，ToriiGate 0.5 要它）→ 启动器的 vlm_venv → ComfyUI 自己的 Python
启动器 / 训练器的位置：config.json → 环境变量 WWY_LAUNCHER_DIR / WWY_LORA_TRAINER → 启动器配置里的 lora_trainer_root。
"""
import json
import os
import sys

PLUGIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PLUGIN_DIR, "config.json")


def config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _models_dir():
    try:
        import folder_paths
        return folder_paths.models_dir
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(PLUGIN_DIR)), "models")


def comfy_vlm_dir():
    return os.path.join(_models_dir(), "vlm_caption")


def comfy_wd14_dir():
    return os.path.join(_models_dir(), "wd14_tagger")


def sibling_plugin(name):
    """custom_nodes 里别的插件的目录（找不到返回 ""）。"""
    dirs = [os.path.dirname(PLUGIN_DIR)]
    try:
        import folder_paths
        dirs += folder_paths.get_folder_paths("custom_nodes")
    except Exception:
        pass
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.isdir(p):
            return p
    return ""


def launcher_dir():
    d = (config().get("launcher_dir") or os.environ.get("WWY_LAUNCHER_DIR") or "").strip()
    if d and os.path.isfile(os.path.join(d, "launcher_config.json")):
        return d
    return ""


def trainer_root():
    d = (config().get("trainer_root") or os.environ.get("WWY_LORA_TRAINER") or "").strip()
    if not d:
        ld = launcher_dir()
        if ld:
            try:
                with open(os.path.join(ld, "launcher_config.json"), encoding="utf-8") as f:
                    d = (json.load(f).get("lora_trainer_root") or "").strip()
            except (OSError, ValueError, AttributeError):
                d = ""
    if d and os.path.isfile(os.path.join(d, "app", "main.py")):
        return d
    return ""


def launcher_vlm_dir():
    ld = launcher_dir()
    return os.path.join(ld, "vlm_models") if ld else ""


def extra_vlm_dirs():
    return [d for d in (config().get("extra_vlm_dirs") or []) if isinstance(d, str) and d.strip()]


def wd14_dirs(key, repo):
    """WD14 模型可能在的目录（每个目录里是 model.onnx + selected_tags.csv）。"""
    out = []
    ld = launcher_dir()
    if ld:
        out.append(os.path.join(ld, "wd_tagger_models", key))
    tr = trainer_root()
    if tr:
        out.append(os.path.join(tr, "models", "taggers", repo.replace("/", "__")))
    for extra in extra_vlm_dirs():
        out.append(os.path.join(extra, repo.replace("/", "__")))
    out.append(os.path.join(comfy_wd14_dir(), key))
    return out


def use_mirror():
    """下载时 hf-mirror / 魔搭排前面（默认，国内用户多）；config.json 写 "mirror": false 就官方优先。"""
    v = config().get("mirror")
    return True if v is None else bool(v)


def _env_ok(py, need=("torch", "transformers", "PIL")):
    if not py or not os.path.isfile(py):
        return False
    root = os.path.dirname(os.path.dirname(os.path.abspath(py)))
    for sp in (os.path.join(root, "Lib", "site-packages"), os.path.join(os.path.dirname(py), "Lib", "site-packages")):
        if os.path.isdir(sp):
            return all(os.path.isdir(os.path.join(sp, n)) for n in need)
    return False


def vlm_pythons():
    """能跑自然语言模型的 Python，按优先顺序 [(名字, 路径)]。"""
    out = []
    p = (config().get("vlm_python") or "").strip()
    if p and os.path.isfile(p):
        out.append(("config.json 指定的环境", p))
    tr = trainer_root()
    if tr:
        for p in (os.path.join(tr, "env", "Scripts", "python.exe"), os.path.join(tr, "env", "bin", "python")):
            if _env_ok(p):
                out.append(("LoRA 训练器的环境", p))
    ld = launcher_dir()
    if ld:
        for p in (os.path.join(ld, "vlm_venv", "Scripts", "python.exe"), os.path.join(ld, "vlm_venv", "bin", "python")):
            if _env_ok(p):
                out.append(("启动器的反推环境", p))
    out.append(("ComfyUI 自己的环境", sys.executable))
    return out


def ai_api():
    """AI 接口设置 {provider, api_key, base_url, model, format}。
    先读启动器 / 训练器共用的 %APPDATA%\\WWY\\ai_api.json；那边没填 Key（没装启动器）时，读插件 config.json 里的 "api"：
    {"api": {"provider": "deepseek", "api_key": "sk-...", "base_url": "", "model": ""}}，provider 可选 deepseek / qwen / claude / custom。"""
    a = _launcher_ai_api()
    if a["api_key"]:
        return a
    c = config().get("api") or {}
    if not isinstance(c, dict) or not str(c.get("api_key") or "").strip():
        return a
    return {"provider": str(c.get("provider") or "deepseek").strip(), "api_key": str(c.get("api_key") or "").strip(),
            "base_url": str(c.get("base_url") or "").strip(), "model": str(c.get("model") or "").strip(),
            "format": str(c.get("format") or "openai").strip()}


def _launcher_ai_api():
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    try:
        with open(os.path.join(base, "WWY", "ai_api.json"), encoding="utf-8") as f:
            v = (json.load(f) or {}).get("values") or {}
    except (OSError, ValueError, AttributeError):
        v = {}
    return {"provider": (v.get("diag_llm_provider") or "deepseek").strip(), "api_key": (v.get("diag_llm_api_key") or "").strip(),
            "base_url": (v.get("diag_llm_base_url") or "").strip(), "model": (v.get("diag_llm_model") or "").strip(),
            "format": (v.get("diag_llm_format") or "openai").strip()}
