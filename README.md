# comfyui-wwy-caption · WWY 反推

把 [WWY 启动器](https://github.com/wangjue520/forge-webui-launcher)「图片反推」页的那一套做成 ComfyUI 节点：WD14 标签反推、标签筛选、自然语言反推、合并提示词、AI 优化提示词。模型和 AI 接口设置都和启动器、[WWY LoRA 训练器](https://github.com/wangjue520/wwy-lora-trainer) 共用，那边下载过的模型这边不用再下。

**推荐用 WWY 启动器（或整合包）启动 ComfyUI。** 没有启动器也能用，就是要自己配几样东西，见 [没有启动器怎么用.txt](没有启动器怎么用.txt)。

## 节点

都在「WWY 反推」分类下，双击画布搜「反推」就能找到。

| 节点 | 输入 → 输出 | 说明 |
| --- | --- | --- |
| 反推：WD14 标签 | 图像 → 标签 | WD14 / PixAI 标签模型，第一次用自动下载。「通用阈值」越高，标签越少越准 |
| 反推：筛选标签 | 标签 → 标签 | 去掉画师、评分和元标签、角色名、作品名、人物特征（发色瞳色等）；黑名单 / 只保留支持 `*` 通配；「最多几个」只留最有把握的 |
| 反推：自然语言 | 图像（+ 参考标签）→ 描述 | 视觉模型看图写英文描述：ToriiGate 0.5（二次元）、JoyCaption、Qwen3-VL，或 API。接上「参考标签」就是启动器的「标签 + 自然语言」模式。「不描述画风」默认开 |
| 合并提示词 | 主提示词 + 标签 + 描述 → 提示词 | 你写的放最前面，反推标签去重后接上，最后是描述句子。没接的输入直接跳过 |
| AI 优化提示词 | 提示词 → 提示词 | 让大模型把提示词写完整，见下 |

### AI 优化提示词

- **写法**
  - Anima（标签 + 自然语言）：质量词 → 人数和主体标签 → 内容标签 → 2 到 4 句英文描述。
  - 纯自然语言（Krea2 等）：4 到 6 句完整、具体的英文段落。
  - 只翻译整理：只翻成英文、整理格式，不加内容。
  - 写法参考了 [comfyui-mcp](https://www.npmjs.com/package/comfyui-mcp) 的 `anima-base`、`krea2-txt2img` 两个 skill。
- **补外貌 / 补服装 / 补动作表情 / 补背景 / 补光线构图**：关掉哪个，AI 就不新加那一类（比如用了角色 LoRA，关掉补外貌、补服装）。你自己写的照样保留。
- **查字典**：要装 [中文标签补全](https://github.com/wangjue520/comfyui-zh-tag-autocomplete)。写之前把你写的中文查成标准标签交给 AI（「弹吉他」→ `playing guitar`）；Anima 写法写完后逐个核对标签，字典里没有的（AI 编出来的）和画师标签删掉。
- 不会加画师和画风；Anima 写法里，AI 写出来的画风标签会自动删掉。
- 「种子」换个数字就重新写一遍；「额外要求」写具体想法，比如「换成夜晚」。

## 模型和环境

- 自然语言模型在单独的进程里跑，写完就退出、显存马上还回来。环境按这个顺序找：config.json 指定的 → LoRA 训练器的环境 → 启动器的反推环境 → ComfyUI 自己的环境。
- ToriiGate 0.5 要 transformers 5 以上；ComfyUI 自带的 transformers 4.5x 跑不了它，Qwen3-VL 可以。
- WD14 要 `onnxruntime`（或 `onnxruntime-gpu`）。
- 模型找不到时自动下载（hf-mirror / 魔搭 / HF 官方轮换，断点续传）。没有启动器 / 训练器时放在 `ComfyUI/models/wd14_tagger` 和 `ComfyUI/models/vlm_caption`。

## 设置（config.json）

不需要也能用。要改的话把 `config.example.json` 复制一份改名成 `config.json`：

| 项 | 说明 |
| --- | --- |
| `launcher_dir` | WWY 启动器文件夹（也可以用环境变量 `WWY_LAUNCHER_DIR`） |
| `trainer_root` | WWY LoRA 训练器文件夹（也可以用 `WWY_LORA_TRAINER`）；不填时从启动器配置里读 |
| `vlm_python` | 指定跑自然语言模型的 Python |
| `extra_vlm_dirs` | 额外的模型目录 |
| `mirror` | `true`（默认）国内镜像优先；`false` 官方优先 |
| `api` | 没有启动器时的 AI 接口：`provider`（deepseek / qwen / claude / custom）、`api_key`，自定义接口再填 `base_url`、`model` |

装了启动器时，AI 接口以启动器「设置 → 报错诊断」里填的为准（存在 `%APPDATA%\WWY\ai_api.json`，和训练器共用）。

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/wangjue520/comfyui-wwy-caption
```

重启 ComfyUI。

## 许可

MIT
