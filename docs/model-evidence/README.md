# Model evidence for the live seed proposer

Public API responses behind the model facts in [llm-seeds.md](../llm-seeds.md), committed
unchanged as fetched at **2026-10-04T20:18:59Z**. No LLM was called to get them. The same
three files, with the same SHA-256, are committed in the sibling repository asof-research
(`results/forward-2026-09/model-evidence/`).

| File | URL | What it shows | SHA-256 |
|---|---|---|---|
| [hf-qwen38-rev.json](hf-qwen38-rev.json) | `https://huggingface.co/api/models/Qwen/Qwen3.8-27B/revision/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | the pinned revision, `lastModified` 2026-08-14T15:00:01Z, repository `createdAt` 2026-08-05T08:22:59Z, licence apache-2.0 | `888ca751f3379f08cd73b4a6025243c0e73dbcad828fe9691ab30b6c142880d8` |
| [hf-qwen38-commits.json](hf-qwen38-commits.json) | `https://huggingface.co/api/models/Qwen/Qwen3.8-27B/commits/main` | commit history of `main`; the pinned revision is the commit of 2026-08-14T15:00:01Z | `89ce4c5ed75bf376ffc68dc1ede542db4be166d33b92dce7f9aae6039de8330e` |
| [openrouter-qwen38-free.json](openrouter-qwen38-free.json) | `https://openrouter.ai/api/v1/models` (the `qwen/qwen3.8-27b:free` entry) | `created` 2026-08-14, `hugging_face_id` Qwen/Qwen3.8-27B, price 0, reasoning efforts offered: xhigh, medium, low | `2b6ef2b56cdb7bc8570bd6f2c13a57757ed8faf793d8cdd091466798e143e893` |

OpenRouter does not say which revision or quantisation it serves; that the proposer runs these
weights rests on OpenRouter serving this release.
