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
[Jev](https://docs.typesafe.ai), TypeSafe's System One decision model.

- **Compare candidates:** up to 10 boxes (the first line of each box is its name) plus shared context.
  - *Pick one:* a single Choice across all candidates, which returns a probability distribution.
  - *Judge each:* every candidate × every criterion, as Noul (P(yes)) or Score, with a weighted
    composite calculated in code. Each candidate is one request, and all its criteria are answered in parallel.
- **Single question:** a raw playground for Noul / Choice / Score.
- **Request sent / raw response:** the exact JSON, so you can copy it into your own code.

Ships with a *Theories of consciousness* preset (9 theories with sources, plus evidence notes and
criteria) and a small support-ticket preset for sanity checks.

## Keys

Paste each key once per device. It is saved in that browser's `localStorage`, the same scheme as
blankaiarena and openrouter-claude: "Remember my keys on this device" is ticked by default, and
"Forget saved keys" clears them. The Keys panel folds away once a key is saved. Keys are sent with
each request through the Space, and the Space never stores or logs them.

- **TypeSafe Jev (native):** key from [console.typesafe.ai](https://console.typesafe.ai).
- **OpenRouter LLM (logprob emulation):** key from [openrouter.ai](https://openrouter.ai/keys).
  The model list shows OpenRouter models that expose token logprobs. Each one is forced to answer
  with a single label token, and the probability on each label is read off. This is a comparison
  baseline, not a calibrated classifier.

Alternatively, on a **private** duplicate you can set the Space secrets `TYPESAFE_API_KEY` /
`OPENROUTER_API_KEY`. Don't set them on a public Space, because anyone could spend your credit.

## Caveat

Jev judges what is written in the state, using common sense from training. A "which theory is
correct" Choice tells you how the text and general knowledge lean. It is not ground truth.
