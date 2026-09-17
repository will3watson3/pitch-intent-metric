# Optional local model assets

Binary weights and broadcast-logo assets are excluded from Git. The offline example and portable tests do not require them.

Hosted glove detection is configured through `run_roboflow_workflow.py` and requires an authorized `ROBOFLOW_API_KEY`. Pose estimation uses MediaPipe's packaged pose interface; it may obtain its runtime assets on first use. Local broadcast recovery uses PNG templates under `models/fox_zone_logos/`; without templates its FOX classifier returns `is_fox=False`, so that recovery path will not activate. Obtain any external assets independently and retain their source/license information. Inspect `FoxBroadcastClassifier` in `scripts/detect_fox_zone.py` for the template matching implementation.
