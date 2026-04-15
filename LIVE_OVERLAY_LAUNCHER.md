# Live overlay launcher

One-click entrypoints for the realtime overlay MVP:

- Double-click `run-live-demo.command` in Finder
- Or run `./run-live-demo.sh`

Both launch `scripts/live_overlay_demo.py` with:

- repo-local Python: `.venv/bin/python`
- default camera source: `--source 0`

Optional extra flags can still be passed from Terminal, for example:

```bash
./run-live-demo.sh --output-dir live_output_test --max-frames 300
```
