# Intent model audit

Start `python scripts/labeler_server.py --port 8765` and open `http://127.0.0.1:8765/intent-audit/`.

Compare glove targets, selectable inferred targets, and actual pitch location beside the pitch video. Review model plausibility as `plausible`, `uncertain`, or `wrong`, with notes. The page reads local audited intent predictions and variant predictions; reviews persist in `data/intent_model_audit_labels.csv`.

HTML/CSS/JavaScript has no build step. Model estimates are provisional. See [model details](../docs/MODEL.md) and [pilot reproduction](../docs/WORKFLOWS.md).
