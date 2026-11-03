# What was actually validated

This package was executed on CPU with Python 3.12.14, PyTorch 2.15.0+cpu, NumPy 2.3.5 and float32. CUDA was unavailable. The environment fingerprint is in `validation_machine.json`; this is not a CUDA compatibility certificate.

## Automated tests: 7 passed

1. Linear/stochastic bridge endpoints and finite velocities.
2. Nonempty prediction-independent completion masks on held-out cameras.
3. Score-corrected SDE preserves a known Gaussian endpoint variance within numerical/sampling tolerance.
4. Decoder bounds, reference-renderer gradients and exact teacher decoder ceiling.
5. Object-disjoint splits, train-only normalization and no target-field access in conditioning.
6. Every objective has finite gradients; learned-rollout anchor and its actual renderer loss have gradients into the core; a fixed paired bridge has no parameter dependency.
7. Resuming at a saved checkpoint boundary reproduces uninterrupted CPU model updates exactly.

The actual test transcript is in `validation_tests.txt`.

## End-to-end CPU smoke execution: passed

Executed the fixed `configs/smoke.json` with all controls:

```bash
python -m gs_meeting.cli make-data --config configs/smoke.json
python -m gs_meeting.cli suite --config configs/smoke.json --device cpu --nfe 4 --samples 2 --split test --extra-controls --max-hours 0.15
```

Eight models were trained for 12 optimizer updates each: A0, A1, A2, B0, B1, B2, B0_anchor, B3_anchor. Nine variants were evaluated because B3 reuses A1 weights. Data has 4 training, 2 validation and 2 test identities, 16 anchors and 16 px images. Each evaluation used two generated samples. Generated checkpoint manifests, raw per-object CSVs, plots, qualitative maps and turntables are included in `runs/smoke`. Their report is prominently marked **DEBUG / PIPELINE CHECK ONLY**.

The data audit found mean disoccluded foreground fractions of approximately 26.1% train, 33.3% validation and 30.6% test. The learned anchor was active on 75% of the B2 logged smoke updates with zero tolerance. These checks show that both the completion mask and regularizer execute; they are not evidence for the hypothesis.

An initial transparent two-sided renderer was rejected because completion masks were empty. The final one-sided normal-culling surface proxy is documented explicitly in README/METHODS/DATA_CONTRACT. Its geometry is an approximation and must not be described as a faithful DreamLifting or ordinary two-sided 3DGS renderer.

## Not completed

- No full procedural pilot/convergence experiment, GPU timing, CUDA rasterizer integration or pretrained DreamLifting/LGAA integration.
- No real-data PBR reconstruction, mesh surface error or perceptual-region metric validation.
- LPIPS is optional and was unavailable in the default validation environment; CSV cells are blank.
- No three-seed scientific replication, hypothesis acceptance or novelty validation from the included short run.

The agent should adapt the environment/data/backend, profile the machine, run the pilot, and then collect the meeting results using `AGENT_HANDOFF.md`.
