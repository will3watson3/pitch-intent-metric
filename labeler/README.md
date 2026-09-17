# Catcher target labeler

Start `python scripts/labeler_server.py --port 8765` from the repository root and open `http://127.0.0.1:8765/`.

Label setup location, confidence, visibility, and notes for pitches in `data/*labeling_queue*.csv`. The server persists labels in local data files. `index.html`, `app.js`, and `styles.css` are served directly without a build step. Real review inputs are not bundled; see [workflows](../docs/WORKFLOWS.md).
