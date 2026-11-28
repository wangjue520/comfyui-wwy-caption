# -*- coding: utf-8 -*-
"""
跑自然语言模型要多新的 transformers，以及「插件自带的 transformers」。

ToriiGate 0.5 是 Qwen3.5 架构（model_type = qwen3_5），transformers 5 才认识；ComfyUI 自带的多半还是 4.5x。
直接升级 ComfyUI 的 transformers 可能弄坏 ComfyUI 本身或别的插件，所以这里不动它：
  用 ComfyUI 自己的 Python，把 transformers 5 和它要的新依赖装进插件目录下单独的文件夹（pip --target），
  只有反推子进程启动时把这个文件夹放到 sys.path 最前面。torch / numpy / PIL 照样用 ComfyUI 的，不重复下载。

装什么由 pip 自己算：先在 ComfyUI 的环境里 --dry-run（会参考已经装了的包），
只把缺的、版本不够的装进去（一般是 transformers、huggingface_hub、tokenizers 和几个纯 Python 的小包，几十 MB）。
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

from . import locate

LIBS_DIR = os.path.join(locate.PLUGIN_DIR, "_vlm_libs")   # 插件自带的 transformers（.gitignore 里）
MARK = "wwy_libs.json"
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
PIP_MIRRORS = ["https://pypi.tuna.tsinghua.edu.cn/simple", "https://mirrors.aliyun.com/pypi/simple/"]


class EnvError(Exception):
    pass


def vtuple(s):
    """'5.18.0' / '5.0.0.dev0' / '4.57.1+cu' → (5, 18, 0)；认不出来返回 ()。"""
    out = []
    for p in re.split(r"[.+-]", str(s or "").strip()):
        m = re.match(r"\d+", p)
        if not m:
            break
        out.append(int(m.group()))
        if m.group() != p:
            break
    return tuple(out)


def at_least(have, need):
    if not need:
        return True
    h, n = vtuple(have), vtuple(need)
    if not h:
        return False
    w = max(len(h), len(n))
    return h + (0,) * (w - len(h)) >= n + (0,) * (w - len(n))


def _dist_version(sp, name):
    """site-packages 里某个包的版本（看 xxx-1.2.3.dist-info 目录，不用启动那个 Python）。"""
    if not sp or not os.path.isdir(sp):
        return ""
    want = re.sub(r"[-_.]+", "_", name).lower()
    best = ""
    try:
        names = os.listdir(sp)
    except OSError:
        return ""
    for n in names:
        if not n.endswith(".dist-info"):
            continue
        base = n[:-len(".dist-info")]
        pkg, _, ver = base.rpartition("-")
        if re.sub(r"[-_.]+", "_", pkg).lower() == want and vtuple(ver) > vtuple(best):
            best = ver
    return best


def _same_py(a, b):
    try:
        return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
    except OSError:
        return False


def pkg_version(py, name, extra=""):
    """py 这个 Python（再加上 extra 文件夹，放在前面）里 name 包的版本，没装返回 ""。"""
    v = _dist_version(extra, name) if extra else ""
    if v:
        return v
    if _same_py(py, sys.executable):
        try:
            from importlib import metadata
            return metadata.version(name)
        except Exception:
            pass
    return _dist_version(locate.site_packages(py), name)


# ============================================================
# 插件自带的 transformers
# ============================================================
def _libs_info():
    try:
        with open(os.path.join(LIBS_DIR, MARK), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _py_tag(py):
    try:
        r = subprocess.run([py, "-I", "-c", "import sys; print('%d.%d' % sys.version_info[:2])"], capture_output=True,
                           text=True, timeout=60, creationflags=_NO_WINDOW)
        return r.stdout.strip()
    except Exception:
        return ""


def libs_for(py, need):
    """已经装好、给 py 用的插件自带 transformers 目录（版本够 need）；没有返回 ""。"""
    info = _libs_info()
    if not info or not os.path.isdir(LIBS_DIR):
        return ""
    if not _same_py(info.get("python") or "", py):
        return ""
    if not at_least(_dist_version(LIBS_DIR, "transformers"), need):
        return ""
    return LIBS_DIR


def _pip_sources():
    official = []   # 空 = pip 自己的设置（用户的 pip.ini 或 PyPI 官方）
    if locate.use_mirror():
        return [["-i", u] for u in PIP_MIRRORS] + [official]
    return [official] + [["-i", u] for u in PIP_MIRRORS]


def _run(cmd, log, timeout=1800):
    """跑 pip，输出一行行写进日志，返回 (退出码, 最后几行)。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONUTF8="1")
    for k in ("PYTHONHOME", "PYTHONPATH", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(k, None)
    tail = []
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, creationflags=_NO_WINDOW)
    except OSError as e:
        return -1, [str(e)]
    for raw in p.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if not line:
            continue
        tail = (tail + [line])[-12:]
        if log and not line.startswith(("  ", "Requirement already satisfied")):
            log("  pip: " + line[:200])
    try:
        return p.wait(timeout), tail
    except subprocess.TimeoutExpired:
        p.kill()
        return -1, tail + ["超时"]


def install_libs(py, need, log=None):
    """给 py 这个 Python 装插件自带的 transformers（版本 >= need，< 6），返回目录。失败抛 EnvError。"""
    log = log or (lambda *_a: None)
    have = pkg_version(py, "transformers")
    log(f"这个模型要 transformers {need} 以上，能用的环境里都没有（ComfyUI 的是 {have or '没装'}）。"
        f"不改 ComfyUI 本身，给反推单独装一份到插件目录（只装一次，几十 MB）…")
    if subprocess.run([py, "-m", "pip", "--version"], capture_output=True, creationflags=_NO_WINDOW).returncode != 0:
        raise EnvError(f"这个 Python 里没有 pip，没法自动装 transformers：{py}")
    specs = [f"transformers>={need},<6"]
    if pkg_version(py, "accelerate"):
        specs.append("accelerate>=1.1.0")   # transformers 5 要新版 accelerate；没装 accelerate 就不装（照样能加载）

    # 1) dry-run：参考已经装了的包，算出要装哪些、装什么版本
    rep_path = os.path.join(tempfile.gettempdir(), f"wwy_vlm_pip_report_{os.getpid()}.json")
    plan, errs = None, []
    for src in _pip_sources():
        if os.path.exists(rep_path):
            os.remove(rep_path)
        rc, tail = _run([py, "-m", "pip", "install", "--dry-run", "--quiet", "--no-warn-script-location",
                         "--report", rep_path] + src + specs, None, timeout=600)
        if rc == 0 and os.path.isfile(rep_path):
            try:
                with open(rep_path, encoding="utf-8") as f:
                    plan = [(x["metadata"]["name"], x["metadata"]["version"]) for x in json.load(f).get("install") or []]
                break
            except (OSError, ValueError, KeyError, TypeError) as e:
                errs.append(f"读 pip 的结果失败：{e}")
        else:
            errs.append(" / ".join(src[1:]) or "默认源")
            errs.append("\n".join(tail[-4:]))
    try:
        os.remove(rep_path)
    except OSError:
        pass
    if plan is None:
        raise EnvError("算不出要装哪些包（pip 太旧要 22.2 以上，或者连不上 PyPI / 镜像）：\n" + "\n".join(errs[-4:]))
    if not any(re.sub(r"[-_.]+", "_", n).lower() == "transformers" for n, _ in plan):
        raise EnvError(f"pip 没给出能装的 transformers {need}（这个 Python 版本可能太旧：transformers 5 要 Python 3.10 以上）")
    pins = [f"{n}=={v}" for n, v in plan]
    log("要装：" + ", ".join(pins))

    # 2) 装进临时文件夹，装好了再换上去（装到一半失败不会留下坏的）
    tmp = LIBS_DIR + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    ok, tail = False, []
    for src in _pip_sources():
        rc, tail = _run([py, "-m", "pip", "install", "--no-deps", "--no-warn-script-location", "--target", tmp] + src + pins,
                        log)
        if rc == 0:
            ok = True
            break
        shutil.rmtree(tmp, ignore_errors=True)
    if not ok:
        raise EnvError("transformers 安装失败。pip 最后的输出：\n" + "\n".join(tail[-8:]))
    shutil.rmtree(os.path.join(tmp, "bin"), ignore_errors=True)   # 命令行脚本用不到
    with open(os.path.join(tmp, MARK), "w", encoding="utf-8") as f:
        json.dump({"python": os.path.abspath(py), "py": _py_tag(py), "need": need, "packages": pins}, f, ensure_ascii=False)
    shutil.rmtree(LIBS_DIR, ignore_errors=True)
    os.replace(tmp, LIBS_DIR)
    log(f"装好了：transformers {_dist_version(LIBS_DIR, 'transformers')}（{LIBS_DIR}）")
    return LIBS_DIR
