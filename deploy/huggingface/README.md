---
title: criptolex-cl
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Chilean crypto regulation, answered with citations
---

# criptolex-cl — live demo

Ask a question about Chilean crypto, fintech, AML or tax rules and get an answer
grounded in official sources (BCN, CMF, UAF, SII), with the document, article and
link it came from — or an explicit refusal when the corpus does not support it.

This Space is a thin wrapper around the image built from
[DaslafTorres/criptolex-cl](https://github.com/DaslafTorres/criptolex-cl), where
the source code, the golden-set evaluation and the design notes live.

- Answers are written in Spanish because the corpus is; the interface toggles EN/ES.
- `GROQ_API_KEY` is configured as a Space secret, never in the repository.
- Informational only — not legal or tax advice.

## Setting it up

1. Create a Space with **SDK: Docker**, hardware **CPU basic** (free).
2. Add these two files to the Space repository root: this `README.md` and the
   `Dockerfile` next to it.
3. Add `GROQ_API_KEY` under **Settings → Variables and secrets** (secret).
4. Make the container package public once: GitHub → your profile → Packages →
   `criptolex-cl` → Package settings → Change visibility → Public. Without this
   the Space build cannot pull the image.
