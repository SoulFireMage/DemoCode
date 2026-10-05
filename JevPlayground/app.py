"""Jev Playground: put things in boxes, ask typed questions, read the probabilities.

Bring your own key. Keys are used for the request you make and are never stored
or logged. On a private duplicate of this Space you can instead set the secrets
TYPESAFE_API_KEY / OPENROUTER_API_KEY.
"""

from __future__ import annotations

import html
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from backends import BACKENDS, BackendError, OpenRouterEmulation, TypeSafeBackend
from presets import PRESETS, SCORE_LEVELS_DEFAULT

MAX_CANDIDATES = 10
TS_NAME, OR_NAME = TypeSafeBackend.name, OpenRouterEmulation.name
ENV_KEYS = {TS_NAME: "TYPESAFE_API_KEY", OR_NAME: "OPENROUTER_API_KEY"}
DEFAULT_PRESET = "Theories of consciousness"


# --- helpers -------------------------------------------------------------------

def resolve_key(backend: str, key: str) -> str:
    return (key or "").strip() or os.environ.get(ENV_KEYS[backend], "")


def parse_candidates(boxes, n):
    out, seen = [], set()
    for text in boxes[: int(n)]:
        text = (text or "").strip()
        if not text:
            continue
        name, _, desc = text.partition("\n")
        name = name.strip()[:80] or f"Candidate {len(out) + 1}"
        base, k = name, 2
        while name in seen:
            name, k = f"{base} ({k})", k + 1
        seen.add(name)
        out.append({"name": name, "description": desc.strip()})
    return out


def parse_criteria(text):
    """Lines of 'question' or 'weight | question'."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*\|\s*(.+)$", line)
        w, q = (float(m.group(1)), m.group(2).strip()) if m else (1.0, line)
        out.append({"weight": w, "question": q})
    return out


def parse_options(text):
    """Lines of 'name' or 'name: description' -> Choice criteria map."""
    crit = {}
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        name, sep, desc = line.partition(":")
        crit[name.strip()] = desc.strip() if sep and desc.strip() else None
    return crit


def redact(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def pct(x):
    return f"{100 * x:.0f}%"


def bars_html(title, probs: dict, note=""):
    rows = []
    for k, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        rows.append(
            f'<div class="jp-row"><div class="jp-label">{html.escape(str(k))}</div>'
            f'<div class="jp-track"><div class="jp-fill" style="width:{100 * p:.1f}%"></div></div>'
            f'<div class="jp-num">{pct(p)}</div></div>'
        )
    return (f'<div class="jp-card"><div class="jp-title">{html.escape(title)}</div>'
            f'{"".join(rows)}<div class="jp-note">{note}</div></div>')


def cell(v, txt):
    return f'<td class="jp-cell" style="background:rgba(229,81,186,{0.08 + 0.72 * v:.2f})">{txt}</td>'


# --- actions -------------------------------------------------------------------

def refresh_models(backend, key):
    try:
        models = BACKENDS[backend].list_models(resolve_key(backend, key))
    except Exception as e:
        raise gr.Error(f"Could not list models: {e}")
    default = "jev-latest" if "jev-latest" in models else (models[0] if models else None)
    info = (f"{len(models)} models expose token logprobs, so they can imitate a classifier."
            if backend == OR_NAME else "Every TypeSafe model is a System One classifier.")
    return gr.update(choices=models, value=default, info=info)


def load_preset(name):
    p = PRESETS[name]
    cands = list(p["candidates"])[:MAX_CANDIDATES]
    boxes = cands + [""] * (MAX_CANDIDATES - len(cands))
    return [p["context"], len(cands), p["criteria"], p["pick"], *boxes]


def show_boxes(n):
    return [gr.update(visible=i < int(n)) for i in range(MAX_CANDIDATES)]


def run_pick(backend, key, model, context, n, question, *boxes):
    cands = parse_candidates(boxes, n)
    if len(cands) < 2:
        raise gr.Error("Fill in at least two candidate boxes.")
    q = {"type": "choice", "instructions": question.strip(),
         "criteria": {c["name"]: c["description"] or None for c in cands}}
    state = {"shared_context": context.strip()} if context.strip() else "Choose among the options."
    request = {"state": state, "model": model, "questions": {"pick": q}}
    try:
        res = BACKENDS[backend].ask(resolve_key(backend, key), model, state, {"pick": q})
    except BackendError as e:
        raise gr.Error(str(e))
    a = res["answers"]["pick"]
    note = (f"Top choice <b>{html.escape(a['choice'])}</b> · confidence {a['confidence']:.2f} · "
            f"model {html.escape(res['model'])} · usage {html.escape(json.dumps(res['usage']))}")
    return bars_html(question, a["probabilities"], note), redact(request), redact(res)


def run_matrix(backend, key, model, context, n, criteria_text, kind, levels_text, *boxes):
    cands = parse_candidates(boxes, n)
    crits = parse_criteria(criteria_text)
    if not cands or not crits:
        raise gr.Error("Need at least one candidate and one criterion.")
    levels = [l.strip() for l in levels_text.splitlines() if l.strip()]
    if kind == "Score" and not 2 <= len(levels) <= 10:
        raise gr.Error("A Score needs 2-10 levels, one per line.")

    def q_for(c):
        if kind == "Score":
            return {"type": "score", "instructions": c["question"], "criteria": levels}
        return {"type": "noul", "instructions": c["question"]}

    questions = {f"c{i + 1}": q_for(c) for i, c in enumerate(crits)}
    key = resolve_key(backend, key)

    def one(cand):
        state = {"candidate": cand}
        if context.strip():
            state["shared_context"] = context.strip()
        # Every criterion goes in one request: Jev reads the state once and answers in parallel.
        return state, BACKENDS[backend].ask(key, model, state, questions)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(one, cands))
    except BackendError as e:
        raise gr.Error(str(e))

    def norm(a):
        if a["type"] == "noul":
            return a["noul"], pct(a["noul"])
        top = len(levels) - 1
        return a["score"] / top, f'{a["score"]:.2f}<span class="jp-sub">/{top}</span>'

    total_w = sum(max(c["weight"], 0) for c in crits)
    rows = []
    for cand, (_, res) in zip(cands, results):
        vals = [norm(res["answers"][f"c{i + 1}"]) for i in range(len(crits))]
        comp = (sum(max(c["weight"], 0) * v for c, (v, _) in zip(crits, vals)) / total_w
                if total_w else None)
        rows.append((cand["name"], vals, comp))
    rows.sort(key=lambda r: -(r[2] if r[2] is not None else 0))

    head = "".join(f'<th title="{html.escape(c["question"])}">C{i + 1}<span class="jp-sub">×{c["weight"]:g}</span></th>'
                   for i, c in enumerate(crits))
    body = "".join(
        f'<tr><th class="jp-rowhead">{html.escape(name)}</th>'
        + "".join(cell(v, t) for v, t in vals)
        + (cell(comp, f"<b>{pct(comp)}</b>") if comp is not None else "<td>–</td>") + "</tr>"
        for name, vals, comp in rows
    )
    legend = "".join(f"<li><b>C{i + 1}</b> (weight {c['weight']:g}) {html.escape(c['question'])}</li>"
                     for i, c in enumerate(crits))
    unit = "P(yes)" if kind != "Score" else f"expected level, 0-{len(levels) - 1}"
    table = (f'<div class="jp-card"><div class="jp-title">Each candidate × each criterion ({unit})</div>'
             f'<div class="jp-scroll"><table class="jp-table"><thead><tr><th></th>{head}<th>Composite</th></tr></thead>'
             f'<tbody>{body}</tbody></table></div><ol class="jp-legend">{legend}</ol>'
             f'<div class="jp-note">Composite = weighted mean, computed in code (weight 0 = shown, not counted). '
             f'Change weights and re-run, or do it by eye: the raw judgments do not depend on the weights.</div></div>')

    first_state, first_res = results[0]
    request = {"state": first_state, "model": model, "questions": questions}
    all_res = {c["name"]: r for c, (_, r) in zip(cands, results)}
    return table, redact(request) + f"\n\n// ...and {len(cands) - 1} more like this, one per candidate", redact(all_res)


def run_single(backend, key, model, state_text, qtype, instructions, options_text, t_true, t_false):
    try:
        state = json.loads(state_text)
    except (json.JSONDecodeError, TypeError):
        state = state_text
    if qtype == "Noul":
        q = {"type": "noul", "instructions": instructions}
        if t_true.strip() or t_false.strip():
            q["criteria"] = {"true": t_true.strip() or None, "false": t_false.strip() or None}
    elif qtype == "Choice":
        crit = parse_options(options_text)
        if len(crit) < 2:
            raise gr.Error("A Choice needs at least two options, one per line.")
        q = {"type": "choice", "instructions": instructions, "criteria": crit}
    else:
        levels = [l.strip() for l in options_text.splitlines() if l.strip()]
        if not 2 <= len(levels) <= 10:
            raise gr.Error("A Score needs 2-10 levels, one per line, lowest first.")
        q = {"type": "score", "instructions": instructions, "criteria": levels}
    request = {"state": state, "model": model, "questions": {"q": q}}
    try:
        res = BACKENDS[backend].ask(resolve_key(backend, key), model, state, {"q": q})
    except BackendError as e:
        raise gr.Error(str(e))
    a = res["answers"]["q"]
    if a["type"] == "noul":
        out = bars_html(instructions, {"yes": a["noul"], "no": 1 - a["noul"]},
                        f"noul = {a['noul']:.3f} (probability the answer is yes)")
    elif a["type"] == "choice":
        out = bars_html(instructions, a["probabilities"], f"confidence {a['confidence']:.2f}")
    else:
        probs = {f"{k}: {a['legend'][k]}": v for k, v in a["probabilities"].items()}
        out = bars_html(instructions, probs, f"score {a['score']:.2f} · confidence {a['confidence']:.2f}")
    return out, redact(request), redact(res)


def single_type_changed(t):
    return (gr.update(visible=t != "Noul",
                      label="Options: one per line, 'name: description'" if t == "Choice"
                      else "Levels: one per line, lowest first (2-10)"),
            gr.update(visible=t == "Noul"))


# --- UI ------------------------------------------------------------------------

CSS = """
.jp-card{border:1px solid var(--border-color-primary);border-radius:10px;padding:14px 16px}
.jp-title{font-weight:600;margin-bottom:10px}
.jp-row{display:grid;grid-template-columns:minmax(90px,32%) 1fr 44px;gap:10px;align-items:center;margin:5px 0}
.jp-label{font-size:.9em;overflow-wrap:anywhere}
.jp-track{background:var(--background-fill-secondary);border-radius:4px;height:14px;overflow:hidden}
.jp-fill{background:#E551BA;height:100%}
.jp-num{text-align:right;font-variant-numeric:tabular-nums;font-size:.9em}
.jp-note{margin-top:10px;font-size:.85em;color:var(--body-text-color-subdued)}
.jp-scroll{overflow-x:auto}
.jp-table{border-collapse:separate;border-spacing:3px;font-variant-numeric:tabular-nums;border:none!important;margin:0!important}
.jp-table th,.jp-table td{border:none!important}
.jp-table th{font-weight:600;font-size:.85em;padding:4px 6px;text-align:center}
.jp-table .jp-rowhead{text-align:left;max-width:220px}
.jp-cell{text-align:center;padding:6px 10px;border-radius:4px;color:var(--body-text-color)}
.jp-sub{font-size:.75em;opacity:.7;margin-left:2px}
.jp-legend{font-size:.85em;margin:10px 0 0 18px}
"""

INTRO = """
# Jev Playground
Put things in boxes, ask **typed questions**, get **probabilities** back.
[Jev](https://docs.typesafe.ai) (TypeSafe) is a *System One* model: it doesn't write text, it answers
`noul` (P(yes)), `choice` (distribution over your options) or `score` (position on your rubric).
Bring your own key — it is used for your request only and never stored.
"""

HOW = """
### Two views of the same thing

**Fast (pattern):** *state* = what's on the table. *question* = the judgment. Jev reads state once,
answers every question in parallel, returns numbers not prose. Composite = your weights × its
judgments, done in code. Change weights ≠ re-ask.

**Step by step:**
1. **State** is the material. Here each candidate becomes `{"candidate": {...}, "shared_context": "..."}`.
2. **Questions** are typed:
   - **Noul** → one number in [0, 1], the probability the answer is *yes*. 0.5 means "as likely yes as no",
     **not** "half true".
   - **Choice** → picks one of your options and gives a probability for each (they sum to 1).
   - **Score** → probabilities over ordered levels; `score` is the probability-weighted level.
3. **Confidence** (Choice/Score) measures how concentrated the distribution is. It is *not* a
   probability of being right.
4. Questions refer to parts of the state by name in backticks, e.g. `` `candidate` ``.
5. The *Request sent* panel shows the exact JSON, so you can copy it into your own code.

### What the numbers can and can't mean
Jev judges what is **written in the state**, using common sense learned in training. Asking
"which theory of consciousness is correct?" gets you a calibrated reading of *how the text and
general knowledge lean* — not ground truth about consciousness, which nobody has. The more useful
move is the **criteria matrix**: narrow, checkable questions ("does it make a falsifiable
prediction?", "is it consistent with finding X?") where the evidence is in the state, combined
with weights *you* choose and can argue about.

### Backends
- **TypeSafe Jev (native)** — the real classifier at `api.typesafe.ai`. Key from
  [console.typesafe.ai](https://console.typesafe.ai). The model list comes from `GET /v1/models`.
- **OpenRouter LLM (logprob emulation)** — any OpenRouter chat model that exposes token
  logprobs, forced to answer with one label token; we read the probability on each label.
  Good for comparison; ordinary LLMs aren't trained for calibrated decisions, and each question is a
  separate call. Note: OpenRouter's `typesafe/jev-router` is a *router* that picks an LLM for a chat
  request — it uses Jev internally but does not expose Jev's typed answers, so it isn't listed here.
"""


with gr.Blocks(title="Jev Playground") as demo:
    gr.Markdown(INTRO)
    with gr.Row(equal_height=True):
        backend = gr.Radio(list(BACKENDS), value=TS_NAME, label="Backend", scale=2)
        key = gr.Textbox(label="API key", type="password", scale=2,
                         placeholder="Pasted key is used for this request only")
        model = gr.Dropdown(TypeSafeBackend.fallback_models, value="jev-latest", label="Model",
                            allow_custom_value=True, scale=2,
                            info="Every TypeSafe model is a System One classifier.")
        refresh = gr.Button("↻ List models", scale=1)

    with gr.Tabs():
        with gr.Tab("Compare candidates"):
            with gr.Row():
                preset = gr.Dropdown(list(PRESETS), value=DEFAULT_PRESET, label="Preset", scale=3)
                load = gr.Button("Load preset", scale=1)
            context = gr.Textbox(label="Shared context (evidence, definitions) — referenced as `shared_context`",
                                 lines=5)
            n = gr.Slider(2, MAX_CANDIDATES, step=1, value=len(PRESETS[DEFAULT_PRESET]["candidates"]),
                          label="Number of candidate boxes")
            boxes = []
            with gr.Row():
                cols = [gr.Column(), gr.Column()]
            for i in range(MAX_CANDIDATES):
                with cols[i % 2]:
                    boxes.append(gr.Textbox(label=f"Candidate {i + 1} — first line is its name",
                                            lines=5, max_lines=14))

            with gr.Accordion("① Pick one — a Choice across all candidates", open=True):
                pick_q = gr.Textbox(label="Question", lines=2)
                pick_btn = gr.Button("Ask", variant="primary")
                pick_out = gr.HTML()
            with gr.Accordion("② Judge each candidate against criteria — composite scoring", open=True):
                criteria = gr.Textbox(label="Criteria: one per line, optional 'weight | ' prefix. "
                                            "Refer to `candidate` and `shared_context`.", lines=6)
                with gr.Row():
                    kind = gr.Radio(["Noul (yes/no probability)", "Score"], value="Noul (yes/no probability)",
                                    label="Answer type")
                    levels = gr.Textbox(SCORE_LEVELS_DEFAULT, label="Score levels (lowest first)", lines=5,
                                        visible=False)
                matrix_btn = gr.Button("Evaluate all", variant="primary")
                matrix_out = gr.HTML()
            with gr.Accordion("Request sent / raw response", open=False):
                with gr.Row():
                    req_view = gr.Code(language="json", label="Request sent (key not included)")
                    res_view = gr.Code(language="json", label="Raw response")

        with gr.Tab("Single question"):
            with gr.Row():
                with gr.Column():
                    s_state = gr.Textbox(label="State — plain text, or JSON", lines=8,
                                         value="Help! My payouts have been failing for 3 days.")
                    s_type = gr.Radio(["Noul", "Choice", "Score"], value="Choice", label="Question type")
                    s_instr = gr.Textbox(label="Instructions", value="Which team should handle this?")
                    s_opts = gr.Textbox(label="Options: one per line, 'name: description'", lines=5,
                                        value="billing: Payments, invoicing, refunds\n"
                                              "technical: Bugs, outages, integrations\n"
                                              "sales: Pricing, upgrades, new accounts")
                    with gr.Row(visible=False) as noul_row:
                        s_true = gr.Textbox(label="What 'yes' means (optional)")
                        s_false = gr.Textbox(label="What 'no' means (optional)")
                    s_btn = gr.Button("Ask", variant="primary")
                with gr.Column():
                    s_out = gr.HTML()
                    s_req = gr.Code(language="json", label="Request sent (key not included)")
                    s_res = gr.Code(language="json", label="Raw response")

        with gr.Tab("How this works"):
            gr.Markdown(HOW)

    # wiring
    common = [backend, key, model]
    refresh.click(refresh_models, [backend, key], model)
    backend.change(refresh_models, [backend, key], model)
    load.click(load_preset, preset, [context, n, criteria, pick_q, *boxes]).then(show_boxes, n, boxes)
    n.change(show_boxes, n, boxes)
    kind.change(lambda k: gr.update(visible=k == "Score"), kind, levels)
    pick_btn.click(run_pick, [*common, context, n, pick_q, *boxes], [pick_out, req_view, res_view])
    matrix_btn.click(lambda *a: run_matrix(*a[:6], "Score" if a[6] == "Score" else "Noul", *a[7:]),
                     [*common, context, n, criteria, kind, levels, *boxes],
                     [matrix_out, req_view, res_view])
    s_type.change(single_type_changed, s_type, [s_opts, noul_row])
    s_btn.click(run_single, [*common, s_state, s_type, s_instr, s_opts, s_true, s_false],
                [s_out, s_req, s_res])
    demo.load(load_preset, preset, [context, n, criteria, pick_q, *boxes]).then(show_boxes, n, boxes)


if __name__ == "__main__":
    demo.launch(css=CSS, theme=gr.themes.Soft())
