# Deployment Notes

This project is designed to be uploaded to a Hugging Face Docker Space.

Recommended flow:

1. Create a Docker Space on Hugging Face.
2. Upload the project contents from the repository root.
3. Add these secrets in Space settings:
   - `HF_TOKEN`
   - `API_BASE_URL`
   - `MODEL_NAME`
4. Wait for the Space to reach `RUNNING`.
5. Verify these endpoints:
   - `/health`
   - `/validate`
   - `/reset`

Helpful commands:

```bash
python3 scripts/check_core_artifact.py
python3 scripts/smoke_test_space.py --base-url https://your-space-name.hf.space
```
