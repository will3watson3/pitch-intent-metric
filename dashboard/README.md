# Research dashboard

Start `python scripts/labeler_server.py --port 8765` and open `http://127.0.0.1:8765/dashboard/`.

Explore the local labeled-target analysis and pitch summaries. This shared research dashboard uses local analysis files; individual pipeline runs also generate their own `data/runs/<run-name>/dashboard.html`. The static HTML/CSS/JavaScript requires no build step. See [workflows](../docs/WORKFLOWS.md).
