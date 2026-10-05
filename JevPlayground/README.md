---
title: Jev Playground
emoji: ⚖️
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Ask Jev typed questions about candidates in boxes
---

# Jev Playground

Put candidates in boxes, ask **typed questions**, get **probabilities** back from
[Jev](https://openrouter.ai/docs/guides/community/jev) and the other *decision models* on
OpenRouter. One OpenRouter key runs everything, billed to your OpenRouter credits.

- **Model picker:** lists live every OpenRouter model whose output type is `decisions`
  (`GET /api/v1/models?output_modalities=decisions`), so everything in it is genuinely a classifier:
  Jev, plus Perplexity, Liquid, Cloudflare, Upstage and others. Requests go to `POST /api/alpha/decisions`.
- **Compare candidates:** up to 10 boxes (the first line of each box is its name) plus shared context.
  - *Pick one:* a single Choice across all candidates, which returns a probability distribution.
  - *Judge each:* every candidate × every criterion, as Noul (P(yes)) or Score, with a weighted
    composite calculated in code. Each candidate is one request, and all its criteria are answered in parallel.
- **Single question:** a raw playground for Noul / Choice / Score.
- **Request sent / raw response:** the exact JSON, so you can copy it into your own code.

Ships with a *Theories of consciousness* preset (9 theories with sources, plus evidence notes and
criteria) and a small support-ticket preset for sanity checks.

## Backends

- **OpenRouter decision models (default):** Jev and its relatives. Some providers leave out
  `confidence` or the full distribution; confidence is then computed from the probabilities where possible.
- **OpenRouter chat LLM (logprob imitation):** any chat model that exposes token logprobs, forced to
  answer with a single label token, with the probability on each label read off. A comparison
  baseline, not a calibrated classifier, and one call per question.
- **TypeSafe direct:** Jev from `api.typesafe.ai`, for people with a TypeSafe key (under *Advanced*).

## Key

Paste your OpenRouter key once per device. It is saved in that browser's `localStorage`, the same
scheme as blankaiarena and openrouter-claude: "Remember my key on this device" is ticked by default,
and "Forget saved key" clears it. The Key panel folds away once a key is saved. The key is sent with
each request through the Space, and the Space never stores or logs it.

Only one password box is visible (the TypeSafe one is tucked under *Advanced*), because phone
password managers treat two side-by-side password boxes as a sign-up form and can block pasting
into the second.

Alternatively, on a **private** duplicate you can set the Space secrets `OPENROUTER_API_KEY` /
`TYPESAFE_API_KEY`. Don't set them on a public Space, because anyone could spend your credit.

## Caveat

The models judge what is written in the state, using common sense from training. A "which theory is
correct" Choice tells you how the text and general knowledge lean. It is not ground truth.
