# Hardening Notes

This package keeps the proven runtime core intact and adds a stronger quality
layer around it.

What is hardened:

- Score-range guarantees are checked repeatedly across seeded episodes
- Invalid-action stagnation termination is verified explicitly
- Inference structured logs are tested for `[START]`, `[STEP]`, and `[END]`
- FastAPI route behavior is tested with a local client
- Submission-critical files are checked before packaging

Recommended command:

```bash
python3 scripts/run_quality_gate.py
```
