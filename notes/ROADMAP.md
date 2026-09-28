# Roadmap

End goal: a mechanistic account of a Qwen-VL-based driving VLM. How does visual evidence
(agents, signals, road geometry) get routed into a driving decision, and which of those
mechanisms are brittle in ways that matter for safety?

Getting there in stages, each building on the last:

## Stage 1: CLIP (vision encoder alone)
Everything downstream sits on a ViT like this one, and LLaVA-1.5's vision tower *is* CLIP ViT-L/14-336.
- [x] 01 Residual lens: CLS → joint space per layer; patch-level spatial maps
- [x] 02 Direct-effect decomposition (Gandelsman et al. 2023): heads/MLPs → label logits, plus ablation check
- [x] 03 Typographic attacks (Goh et al. 2021): which components read the stamped text?
- [ ] 04 TextSpan: describe each late head by the text directions that explain its output variance
      across a dataset (needs an image set; ImageNet val or a driving set such as nuScenes/BDD100K frames)
- [ ] 05 Spatial decomposition: split head contributions over patch positions (which image regions each head reads)
- [ ] 06 Driving concepts: linear probes on CLS/patches for traffic-light state, pedestrian presence, distance bin
- [ ] Repeat 02/03 on ViT-L/14-336 (LLaVA's tower) and SigLIP (llava-interleave's and Qwen's lineage)

## Stage 2: LLaVA (simplest ViT + projector + LM)
Start with `llava-interleave-qwen-0.5b` locally, then confirm on `llava-1.5-7b` on GPU.
- [x] VLM 01 Logit lens on image tokens
- [ ] 02 Projector: do projected image tokens land near word embeddings? (nearest-token map before layer 0)
- [ ] 03 Visual-token ablation by layer: after which layer can image tokens be dropped? (where image→text transfer finishes)
- [ ] 04 Attention knockout from answer position → image tokens, per layer: find the "read" layers
- [ ] 05 Does the stage-1 typographic circuit survive into the LM? Same attacked images, trace through the projector
- [ ] 06 Language prior vs vision: conflicting prompt and image; which layers let text win?

## Stage 3: Qwen-VL (native resolution, merge, M-RoPE, DeepStack)
- [ ] Rerun the stage-2 suite on Qwen2.5-VL-3B; what changes with dynamic resolution and 2×2 merge?
- [ ] Qwen3-VL DeepStack: what do the ViT features injected into early LM layers contribute?
- [ ] How does M-RoPE carry spatial position? (probe for patch x/y from LM residual stream)

## Stage 4: driving model
- [ ] Pick the target checkpoint + eval data (DriveLM / NAVSIM / nuScenes QA or trajectories)
- [ ] Diff base vs fine-tune: which components changed most (weights + activations)?
- [ ] Circuit for one decision, e.g. "stop for red light": patch clean ↔ counterfactual frames
- [ ] Safety failures: typographic/sticker attacks on signs, hallucinated or missed agents,
      text prompt overriding the scene; look for activation-level detectors
- [ ] SAEs / transcoders on image-position residuals (GPU)

## Tooling backlog
- [ ] Attention-pattern capture (`attn_implementation="eager"`)
- [ ] Activation-patching helper over (layer, position-set) grids
- [ ] Map image tokens → pixel boxes for overlays
- [ ] Batched dataset runner (for TextSpan, probes)

## Log
Add dated entries: what you ran, what you saw, what it means, what's next.

