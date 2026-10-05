"""Two ways to ask a typed question and get probabilities back.

TypeSafeBackend  -> the real thing: Jev via api.typesafe.ai/v1/systemone.
OpenRouterEmulation -> an *imitation* of the same contract using any
OpenRouter chat model that exposes token logprobs. The model is forced to
answer with a single label token and we read the probability mass on each
label. Useful for comparison; it is not calibrated the way Jev is trained to be.

Both return answers in TypeSafe's response shape so the UI does not care
which one ran.
"""

from __future__ import annotations

import json
import math
import string
import time
from concurrent.futures import ThreadPoolExecutor

import requests

TYPESAFE_URL = "https://api.typesafe.ai/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
TIMEOUT = 60


class BackendError(Exception):
    pass


def _post_with_retry(url, headers, payload, retries=3):
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt == retries:
                raise BackendError(f"Network error: {e}") from e
            time.sleep(2 ** attempt)
            continue
        if r.status_code in (429, 529) and attempt < retries:
            wait = r.headers.get("retry-after")
            time.sleep(float(wait) if wait and wait.replace(".", "").isdigit() else 2 ** attempt)
            continue
        if r.status_code >= 400:
            raise BackendError(f"HTTP {r.status_code}: {r.text[:800]}")
        return r.json()
    raise BackendError("Gave up after retries")


# --- confidence, as TypeSafe defines it (docs.typesafe.ai/confidence) ---------

def choice_confidence(probs: list[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (n * max(probs) - 1) / (n - 1)))


def score_confidence(probs: list[float]) -> float:
    n = len(probs)
    if n < 2:
        return 1.0
    peak = probs.index(max(probs))
    spread = sum(p * abs(i - peak) for i, p in enumerate(probs))
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, min(1.0, 1 - spread / even)) if even else 1.0


# --- TypeSafe (Jev) ---------------------------------------------------------

class TypeSafeBackend:
    name = "TypeSafe Jev (native)"
    fallback_models = ["jev-latest", "jev-preview", "jev-1.13.0"]

    def list_models(self, key: str) -> list[str]:
        if not key:
            return self.fallback_models
        try:
            r = requests.get(f"{TYPESAFE_URL}/models",
                             headers={"Authorization": f"Bearer {key}"}, timeout=20)
            r.raise_for_status()
            names = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return self.fallback_models
        # Versioned ids are accepted even when the listing only shows aliases.
        return list(dict.fromkeys(names + self.fallback_models))

    def ask(self, key: str, model: str, state, questions: dict) -> dict:
        if not key:
            raise BackendError("A TypeSafe API key is needed (console.typesafe.ai).")
        payload = {"state": state, "model": model, "questions": questions}
        data = _post_with_retry(
            f"{TYPESAFE_URL}/systemone",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload,
        )
        return {"model": data.get("model", model), "answers": data["answers"],
                "usage": data.get("usage", {})}


# --- OpenRouter logprob emulation ---------------------------------------------

def _fmt(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


class OpenRouterEmulation:
    name = "OpenRouter LLM (logprob emulation)"

    def list_models(self, key: str) -> list[str]:
        """Models that can stand in for a classifier: text out, logprobs exposed,
        and no mandatory hidden reasoning (which would eat the one answer token)."""
        r = requests.get(f"{OPENROUTER_URL}/models", timeout=30)
        r.raise_for_status()
        out = []
        for m in r.json()["data"]:
            params = set(m.get("supported_parameters") or [])
            if not {"logprobs", "top_logprobs"} <= params:
                continue
            if (m.get("reasoning") or {}).get("mandatory"):
                continue
            if "text" not in (m.get("architecture") or {}).get("output_modalities", ["text"]):
                continue
            out.append(m["id"])
        return sorted(out)

    def _label_distribution(self, key, model, state, instructions, labels, descriptions):
        lines = [f"{lab}) {desc}" for lab, desc in zip(labels, descriptions)]
        prompt = (
            f"STATE:\n{_fmt(state)}\n\nQUESTION:\n{_fmt(instructions)}\n\n"
            "OPTIONS:\n" + "\n".join(lines) +
            f"\n\nReply with exactly one label from: {', '.join(labels)}. No other text."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a careful classifier. Output a single option label only."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 20,
            "provider": {"require_parameters": True},
        }
        data = _post_with_retry(
            f"{OPENROUTER_URL}/chat/completions",
            {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            payload,
        )
        try:
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError):
            raise BackendError(f"{model} returned no logprobs for this request; try another model.")
        mass = {lab: 0.0 for lab in labels}
        for t in top:
            tok = t["token"].strip().upper()
            if tok in mass:
                mass[tok] += math.exp(t["logprob"])
        total = sum(mass.values())
        if total <= 0:
            raise BackendError(f"{model} put no probability on any option label "
                               f"(top tokens: {[t['token'] for t in top[:5]]}).")
        usage = data.get("usage", {})
        return [mass[lab] / total for lab in labels], usage

    def _answer(self, key, model, state, q):
        t = q["type"]
        if t == "choice":
            opts = list(q["criteria"])
            labels = list(string.ascii_uppercase[: len(opts)])
            descs = [o if q["criteria"][o] in (None, "") else f"{o}: {_fmt(q['criteria'][o])}" for o in opts]
            probs, u = self._label_distribution(key, model, state, q["instructions"], labels, descs)
            pmap = dict(zip(opts, probs))
            return {"type": "choice", "choice": max(pmap, key=pmap.get),
                    "probabilities": pmap, "confidence": choice_confidence(probs)}, u
        if t == "noul":
            crit = q.get("criteria") or {}
            descs = ["Yes" + (f": {_fmt(crit['true'])}" if crit.get("true") else ""),
                     "No" + (f": {_fmt(crit['false'])}" if crit.get("false") else "")]
            probs, u = self._label_distribution(key, model, state, q["instructions"], ["Y", "N"], descs)
            return {"type": "noul", "noul": probs[0]}, u
        if t == "score":
            levels = q["criteria"]
            labels = [str(i) for i in range(len(levels))]
            probs, u = self._label_distribution(key, model, state, q["instructions"], labels,
                                                [_fmt(lv) for lv in levels])
            return {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
                    "legend": {str(i): _fmt(lv) for i, lv in enumerate(levels)},
                    "probabilities": dict(zip(labels, probs)),
                    "confidence": score_confidence(probs)}, u
        raise BackendError(f"Unknown question type {t}")

    def ask(self, key: str, model: str, state, questions: dict) -> dict:
        if not key:
            raise BackendError("An OpenRouter API key is needed (openrouter.ai/keys).")
        # One chat call per question; this is the cost Jev's single-pass fan-out avoids.
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = {qid: pool.submit(self._answer, key, model, state, q) for qid, q in questions.items()}
            answers, inp = {}, 0
            for qid, f in futs.items():
                answers[qid], u = f.result()
                inp += u.get("prompt_tokens", 0)
        return {"model": model, "answers": answers, "usage": {"input_tokens": inp, "calls": len(questions)}}


BACKENDS = {b.name: b for b in (TypeSafeBackend(), OpenRouterEmulation())}
