# -*- coding: utf-8 -*-
"""
在 ComfyUI 里管自然语言模型的常驻进程（照启动器 vlm_caption_service.py 的做法，简化成同步调用）。

模型跑在单独的进程里（训练器的环境优先），不和 ComfyUI 抢 Python 包版本；进程一退出显存就全还回去。
加载前先让 ComfyUI 把自己的模型挪出显存，用完默认结束进程，后面采样时显存够用。
"""
import json
import os
import subprocess
import tempfile
import threading
import time

from . import locate
from . import vlm_core as vc

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_worker.py")
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
TAG = "[WWY反推]"


def log(*a):
    print(TAG, *a, flush=True)


def _interrupted():
    try:
        import comfy.model_management as mm
        return mm.processing_interrupted()
    except Exception:
        return False


def _free_comfy_vram():
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:
        pass


class Service:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.loaded = None        # (key, dir, quant, python)
        self.events = []
        self.cond = threading.Event()
        self.cancel_file = os.path.join(tempfile.gettempdir(), f"wwy_comfy_vlm_cancel_{os.getpid()}.flag")

    # ---------- 进程 ----------
    def _start(self, py):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                   HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", WWY_PARENT_PID=str(os.getpid()))
        for k in ("PYTHONHOME", "PYTHONPATH"):
            env.pop(k, None)
        self.events = []
        p = subprocess.Popen([py, "-I", "-X", "utf8", "-u", WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env=env, creationflags=_NO_WINDOW)
        self.proc = p
        threading.Thread(target=self._read, args=(p,), daemon=True).start()
        return p

    def _read(self, p):
        for raw in p.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line:
                continue
            if line.startswith("@@"):
                try:
                    ev = json.loads(line[2:])
                except ValueError:
                    ev = None
                if ev is not None:
                    if self.proc is p:
                        self.events.append(ev)
                        self.cond.set()
                    continue
            if "Loading weights" in line or "causal_conv1d" in line:   # 进度条刷屏、无关的提示
                continue
            log(line)
        code = p.wait()
        if self.proc is p:
            self.events.append({"ev": "exit", "code": code})
            self.cond.set()

    def _send(self, cmd):
        p = self.proc
        if not p or p.poll() is not None:
            raise vc.CaptionError("模型进程已经退出")
        p.stdin.write((json.dumps(cmd, ensure_ascii=False) + "\n").encode("utf-8"))
        p.stdin.flush()

    def _wait(self, kinds, on_item=None, timeout=None):
        t0 = time.time()
        while True:
            while self.events:
                ev = self.events.pop(0)
                if ev.get("ev") in kinds or ev.get("ev") == "exit":
                    return ev
                if ev.get("ev") == "item" and on_item:
                    on_item(ev)
            if _interrupted() and not os.path.exists(self.cancel_file):
                open(self.cancel_file, "w").close()   # 这一批生成完就停
            self.cond.wait(0.5)
            self.cond.clear()
            if timeout and time.time() - t0 > timeout:
                return {"ev": "timeout"}

    def release(self):
        with self.lock:
            p, self.proc, self.loaded = self.proc, None, None
        if p and p.poll() is None:
            try:
                p.stdin.write(b'{"cmd": "quit"}\n')
                p.stdin.flush()
                p.wait(8)
            except Exception:
                pass
            if p.poll() is None:
                p.kill()
                try:
                    p.wait(10)
                except Exception:
                    pass
            log("模型已卸载，显存已释放")

    # ---------- 加载 ----------
    def ensure(self, key, quant="auto"):
        """把 key 这个模型准备好（没下载先下载），返回 (目录, python)。"""
        info = vc.VLM_MODELS[key]
        tr = locate.trainer_root()
        d, where = vc.find_model(key, tr)
        if not d:
            d = vc.download_dir(key, tr)
            log(f"本地没有「{info['label']}」，开始下载约 {info['size_gb']} GB 到 {d}")
            pbar = _pbar()
            last = [0.0]

            def prog(done, total, name):
                if pbar and total:
                    pbar.update_absolute(int(done * 1000 / total), 1000)
                if time.time() - last[0] > 10:
                    last[0] = time.time()
                    log(f"下载中 {done / 1e9:.2f}/{total / 1e9:.2f} GB  {name}")

            vc.download_repo(info["repo"], d, progress_cb=prog, log_cb=log, cancel_flag=_interrupted)
        want = (key, d, quant)
        if self.loaded and self.loaded[:3] == want and self.proc and self.proc.poll() is None:
            return d
        self.release()
        _free_comfy_vram()
        errs = []
        for name, py in locate.vlm_pythons():
            log(f"用{name}加载「{info['label']}」：{py}")
            self._start(py)
            ev = self._wait(("hello",), timeout=120)
            if ev.get("ev") != "hello":
                errs.append(f"{name}：模型进程没起来（{ev}）")
                self.release()
                continue
            self._send({"cmd": "load", "dir": d, "key": key, "quant": quant, "size_gb": info["size_gb"], "batch": 1})
            ev = self._wait(("loaded", "load_failed"))
            if ev.get("ev") == "loaded":
                self.loaded = (key, d, quant, py)
                log(f"模型已加载（{ev.get('quant')}）")
                return d
            errs.append(f"{name}：{ev.get('error') or ev}")
            self.release()
        hint = ""
        if key == "toriigate":
            hint = "\nToriiGate 0.5 要 transformers 5 以上：装了 WWY LoRA 训练器时会用训练器的环境（在插件的 config.json 里填 trainer_root）。"
        raise vc.CaptionError("自然语言模型加载失败：\n" + "\n".join(errs) + hint)

    # ---------- 生成 ----------
    def run(self, key, quant, items, max_tokens=512, temperature=0.6, unload=True):
        """items: [{"file": 图片路径 或 "", "id": 文字任务的编号, "system", "user"}] → 每项的文字"""
        with self.lock:
            try:
                self.ensure(key, quant)
                if os.path.exists(self.cancel_file):
                    os.remove(self.cancel_file)
                res = {}

                def on_item(ev):
                    k = ev.get("file") or ""
                    res[k] = ev

                self._send({"cmd": "run", "items": items, "max_tokens": max_tokens, "temperature": temperature,
                            "cancel_file": self.cancel_file})
                ev = self._wait(("done",), on_item=on_item)
                if ev.get("ev") == "exit":
                    raise vc.CaptionError(f"模型进程意外退出（{ev.get('code')}），看上面的日志")
                if ev.get("error"):
                    raise vc.CaptionError(ev["error"])
                if ev.get("cancelled"):
                    raise vc.CaptionError("已取消")
                out = []
                for it in items:
                    r = res.get(it.get("file") or it.get("id") or "") or {}
                    if r.get("error"):
                        raise vc.CaptionError(r["error"])
                    out.append(r.get("text") or "")
                return out
            finally:
                if unload:
                    self.release()


def _pbar():
    try:
        import comfy.utils
        return comfy.utils.ProgressBar(1000)
    except Exception:
        return None


SERVICE = Service()
