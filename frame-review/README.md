# Frame and zone review

Start `python scripts/labeler_server.py --port 8765` and open `http://127.0.0.1:8765/frame-review/`.

Inspect extracted setup frames and their glove/zone overlays, correct geometry, and flag detector problems. Each correction belongs to a specific frame. Local manifests, frame images, detections, and calibration records supply the page. The interface has no build step. See [data contracts](../docs/DATA.md) and [overlay tests](../scripts/tests/test_frame_overlay.js).
