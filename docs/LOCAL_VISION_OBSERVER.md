# Local visual interpretation — optional and untrusted

Vision has two distinct layers: **direct visual evidence** (decoded frames, sampled changes, timestamps) and **inferred visual descriptions** (a model's reading of one frame). Do not convert the latter into an authoritative game state, collision clearance, or permission to operate a controller.

The source implementation includes an explicit opt-in adapter for Ollama's `/api/generate` endpoint using the documented `images`, `format` (JSON schema), and `stream=false` request fields. Official API references: [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md), [Structured Outputs](https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx).

## Local video with occasional scene interpretation

```powershell
vera-vision watch --kind window --source "Chiaki" --fps 2 --frames 60 --seconds 120 --local-vision-model "YOUR_INSTALLED_IMAGE_CAPABLE_MODEL" --local-vision-interval 5
```

Nothing calls Ollama unless `--local-vision-model` is specified. **The model is not downloaded or selected automatically.** This source does not prove that any given installed Vera language model supports images. Install and verify an image-capable model independently, or the calls will return sanitized `UNAVAILABLE` observations.

The adapter connects only to `http://127.0.0.1:11434/api/generate` by default. It refuses remote endpoints and credentials in URLs, disables HTTP proxies and redirects, and never logs frames, prompts, raw model text or provider exception details. This constrains Vision's outbound HTTP connection; it cannot guarantee what an independently configured Ollama backend does with the selected model. **Do not select cloud-forwarding model aliases for private footage.**

Every attempted classification emits a distinct `SEMANTIC_OBSERVATION` JSONL event with session ID, decoded-frame sequence, observer-side monotonic timestamp, scene category, visible category list, bounded summary and *uncalibrated model-reported* confidence. Invalid, incomplete or malformed responses become `UNAVAILABLE`, not invented recognition. Model output is **untrusted data**, and `action_authority=false` is permanent.

The CLI's normal `FRAME_OBSERVED` events continue regardless of semantic availability. The sampler rate-limits model inference (five seconds between calls by default); while inference is running, the live decoder's bounded queue discards older frames rather than building a stale backlog.

The current semantic labels are intentionally generic: `gameplay, desktop, menu, outdoor, indoor, unknown` scenes and `person, animal, vehicle, structure, screen, terrain, text, other` visible categories. They do not identify ARK creatures, inventory elements or safe movement.

## Integration into Vera

The local semantic event is one input to a future **qualified Vera runtime consumer**. It is not proof that `vera_core.QualifiedVeraRuntime` is installed, selected for this route, or using those observations. The host must separately bind exact source/provenance, freshness, explicit authorization, and memory retention policy before presenting results as part of Vera's continuous active perception.

No live Chiaki model performance has been measured yet. Tests use a fake loopback provider and verify schema errors, rate-limits, output bounds and refusal to follow redirects. They do not establish accuracy or a functioning local model deployment.
