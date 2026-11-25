// WWY LoRA 堆的界面：每个 LoRA 三个 ComfyUI 自带控件（下拉框 lora_N、数字框 strength_N、开关 on_N），
// 「➕ 加一个 LoRA」「➖ 去掉最后一个」两个按钮，右键菜单删掉 / 上移中间某一个。
// 全用自带控件，点击、拖动、输入都交给 ComfyUI 自己处理，不会碰到 rgthree 自绘控件那种宽度错位的问题。
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TYPE = "WWYLoraStack";
let LORAS = ["None"];

function setLoras(list) {
    if (Array.isArray(list)) LORAS = ["None", ...list.filter((x) => x !== "None")];
}

async function fetchLoras() {
    try {
        const r = await api.fetchApi("/object_info/LoraLoader");
        const j = await r.json();
        setLoras(j.LoraLoader.input.required.lora_name[0]);
    } catch (e) {
        console.warn("[WWY LoRA 堆] 读不到 LoRA 列表", e);
    }
}

function rowsOf(node) {
    const rows = [];
    const ws = node.widgets || [];
    for (const w of ws) {
        const m = /^lora_(\d+)$/.exec(w.name);
        if (!m) continue;
        const i = m[1];
        const s = ws.find((x) => x.name === `strength_${i}`);
        const o = ws.find((x) => x.name === `on_${i}`);
        rows.push({ lora: w.value, strength: s ? s.value : 1, on: o ? o.value : true });
    }
    return rows;
}

function noSave(w) {
    w.serialize = false;
    w.options = w.options || {};
    w.options.serialize = false;
    return w;
}

function setRows(node, rows) {
    for (const w of node.widgets || []) w.onRemove?.();
    node.widgets = [];
    const redraw = () => node.setDirtyCanvas(true, true);
    rows.forEach((r, k) => {
        const i = k + 1;
        const wl = node.addWidget("combo", `lora_${i}`, r.lora ?? "None", redraw, { values: () => LORAS });
        wl.label = `LoRA ${i}`;
        const ws = node.addWidget("number", `strength_${i}`, r.strength ?? 1, redraw,
            { min: -10, max: 10, step: 0.5, step2: 0.05, round: 0.01, precision: 2 });
        ws.label = "　强度";
        const wo = node.addWidget("toggle", `on_${i}`, r.on ?? true, redraw, { on: "用", off: "先不用" });
        wo.label = "　开关";
        // 新前端按「节点 + 控件名」记着值，同名控件重建时会沿用旧值（删掉 / 挪动之后就错位了），所以再赋一次
        wl.value = r.lora ?? "None";
        ws.value = r.strength ?? 1;
        wo.value = r.on ?? true;
    });
    noSave(node.addWidget("button", "➕ 加一个 LoRA", null, () => {
        fetchLoras();
        setRows(node, [...rowsOf(node), { lora: "None", strength: 1, on: true }]);
    }));
    noSave(node.addWidget("button", "➖ 去掉最后一个", null, () => {
        const rs = rowsOf(node);
        if (rs.length) setRows(node, rs.slice(0, -1));
    }));
    const h = node.computeSize()[1];
    node.setSize([Math.max(node.size[0], 320), h]);
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "wwy.LoraStack",
    async setup() {
        await fetchLoras();
    },
    // 按 R 刷新模型列表时一起刷新
    async refreshComboInNodes(defs) {
        const l = defs?.LoraLoader?.input?.required?.lora_name?.[0];
        if (l) setLoras(l);
        else await fetchLoras();
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TYPE) return;

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = created?.apply(this, arguments);
            this.serialize_widgets = true;
            setRows(this, [{ lora: "None", strength: 1, on: true }]);
            return r;
        };

        // 读工作流：控件是按需加的，按存下来的值（每个 LoRA 三个）重新搭一遍
        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = configure?.apply(this, arguments);
            const v = info?.widgets_values;
            if (Array.isArray(v)) {
                const rows = [];
                for (let k = 0; k + 2 < v.length; k += 3) {
                    rows.push({ lora: v[k], strength: v[k + 1], on: v[k + 2] });
                }
                const keep = this.size ? [...this.size] : null;
                setRows(this, rows);
                if (keep) this.setSize([keep[0], Math.max(keep[1], this.computeSize()[1])]);
            }
            return r;
        };

        const menu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const r = menu?.apply(this, arguments);
            const rs = rowsOf(this);
            const name = (x) => `${x.lora}`.split(/[\\/]/).pop();
            if (rs.length) {
                options.push(null, {
                    content: "删掉某个 LoRA",
                    has_submenu: true,
                    submenu: {
                        options: rs.map((x, k) => ({ content: `${k + 1}. ${name(x)}`, callback: () => setRows(this, rs.filter((_, j) => j !== k)) })),
                    },
                }, {
                    content: "把某个 LoRA 往上挪",
                    has_submenu: true,
                    submenu: {
                        options: rs.slice(1).map((x, k) => ({
                            content: `${k + 2}. ${name(x)}`,
                            callback: () => { const a = [...rs]; [a[k], a[k + 1]] = [a[k + 1], a[k]]; setRows(this, a); },
                        })),
                    },
                });
            }
            return r;
        };
    },
});
