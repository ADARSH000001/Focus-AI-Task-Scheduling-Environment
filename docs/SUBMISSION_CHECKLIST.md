# Submission Checklist

- `Dockerfile` is present at repo root
- `inference.py` is present at repo root
- `uv.lock` is present
- `server/app.py` exists
- `pyproject.toml` exposes the server entry point
- `openenv.yaml` is present
- Baseline scores are strictly inside `(0, 1)`
- HF Space responds on `/health`
- HF Space responds on `/validate`
- `POST /reset` returns `200`
- `POST /step` returns `200`

Before submitting, run:

```bash
python3 scripts/check_core_artifact.py
python3 inference.py
```
