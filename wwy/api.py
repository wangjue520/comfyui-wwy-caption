# -*- coding: utf-8 -*-
"""
用启动器 / 训练器设置好的 AI 接口（%APPDATA%\\WWY\\ai_api.json，和「报错诊断」同一份）写提示词、看图反推。
服务商的默认地址和模型照启动器 app/diagnose_llm.py。Key 只从那个文件读，不写进工作流。
"""
import base64
import io
import json
import re
import urllib.error
import urllib.request

from . import locate
from . import vlm_core as vc

PROVIDERS = {
    "deepseek": {"label": "DeepSeek", "format": "openai", "base": "https://api.deepseek.com", "model": "deepseek-v4-flash",
                 "vision": ""},
    "qwen": {"label": "通义千问（阿里云百炼）", "format": "openai", "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "model": "qwen-plus", "vision": "qwen3-vl-plus"},
    "claude": {"label": "Claude", "format": "anthropic", "base": "https://api.anthropic.com", "model": "claude-sonnet-5-5",
               "vision": "claude-sonnet-5-5"},
    "custom": {"label": "自定义接口", "format": "openai", "base": "", "model": "", "vision": ""},
}


# 能看图的模型名：Claude 全系都能看图；其他家看名字（vl / qvq / omni / vision，Qwen3.5 起原生多模态）
_VISION_NAME = re.compile(r"(?:^|[-_./])(?:vl|qvq|omni|vision|visual)(?:$|[-_./\d])|qwen3\.[5-9]|qwen[4-9]|gpt-4o|gpt-[5-9]|gemini|glm-\d+(?:\.\d+)?v",
                          re.I)


def can_see(provider, model):
    if not model:
        return False
    if provider == "claude" or model.lower().startswith("claude"):
        return True
    return bool(_VISION_NAME.search(model))


def resolve(vision=False, model_override=""):
    """返回 (格式, 地址, 模型, Key)。模型按这个顺序选：
    节点上填的「API模型」 → 看图时：设置里单独填的看图模型 → 设置里的模型（能看图才用）→ 服务商默认的看图模型；
    写字时：设置里的模型 → 服务商默认模型。"""
    a = locate.ai_api()
    p = PROVIDERS.get(a["provider"]) or PROVIDERS["deepseek"]
    if not a["api_key"]:
        raise vc.CaptionError("还没设置 AI 接口。装了 WWY 启动器：在「设置 → 报错诊断」里选服务商、填 Key（和训练器共用）。"
                              "没装启动器：打开本插件文件夹里的 config.json（没有就把 config.example.json 复制一份改名），"
                              "在 \"api\" 里填 provider 和 api_key，保存后重新运行。详见「没有启动器怎么用.txt」。")
    fmt = (a["format"] or p["format"]) if a["provider"] == "custom" else p["format"]
    base = (a["base_url"] or p["base"]).rstrip("/")
    model = (model_override or "").strip()
    if not model:
        if vision:
            mine = a.get("model", "")
            if a.get("vision_model"):
                model = a["vision_model"]
            elif mine and (a["provider"] == "custom" or can_see(a["provider"], mine)):
                model = mine
            else:
                model = p["vision"]
            if not model:
                raise vc.CaptionError(f"现在设置的服务商「{p['label']}」不能看图。看图反推请换成通义千问（百炼）或 Claude，"
                                      "或者在节点的「API模型」里填一个能看图的模型名（也可以在 config.json 的 \"api\" 里填 \"vision_model\"）。")
            if mine and model != mine and not a.get("vision_model"):
                print(f"[WWY反推] 设置里的模型 {mine} 看不了图，看图改用 {model}"
                      "（要指定别的，在节点的「API模型」或 config.json 的 \"vision_model\" 里填）", flush=True)
        else:
            model = a["model"] or p["model"]
    if not base or not model:
        raise vc.CaptionError("AI 接口的地址或模型名是空的：去启动器的 AI 设置里补上，或者在插件 config.json 的 \"api\" 里填 base_url 和 model。")
    return fmt, base, model, a["api_key"]


def _post(url, headers, body, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                 headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            e.body = e.read().decode("utf-8", errors="replace")
        except Exception:
            e.body = ""
        raise vc.CaptionError("AI 接口出错：" + vc.api_error_zh(e)) from None
    except Exception as e:
        raise vc.CaptionError("AI 接口出错：" + vc.api_error_zh(e)) from None


def _png_b64(pil):
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def chat(system, user, image=None, max_tokens=800, temperature=0.4, model_override=""):
    fmt, base, model, key = resolve(vision=image is not None, model_override=model_override)
    if fmt == "anthropic":
        url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        content = [{"type": "text", "text": user}]
        if image is not None:
            content.insert(0, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _png_b64(image)}})
        d = _post(url, {"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                  {"model": model, "max_tokens": max_tokens, "system": system, "temperature": temperature,
                   "messages": [{"role": "user", "content": content}]})
        text = "".join(b.get("text", "") for b in d.get("content") or [] if b.get("type") == "text")
        finish = d.get("stop_reason")
        refusal = None
    else:
        content = user
        if image is not None:
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(image)}}]
        d = _post(base + "/chat/completions", {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                  {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]})
        ch = (d.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):
            text = " ".join(x.get("text", "") for x in text if isinstance(x, dict))
        finish, refusal = ch.get("finish_reason"), msg.get("refusal")
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    why = vc.refusal_reason(text, finish, refusal)
    if why:
        raise vc.CaptionError(why)
    return text
