# Testing

Install `requirements-core.txt`, then run:

```sh
python -m unittest discover -s scripts/tests -p 'test_*.py' -v
python examples/synthetic_demo.py
node scripts/tests/test_frame_overlay.js
```

These commands apply to the published checkout, which includes only the portable test suite. A longstanding research workspace may also contain ignored experimental tests requiring local fixtures; use individual `-p test_intent_synthetic.py`, `-p test_zone_refinement.py`, `-p test_frame_payload.py`, and `-p test_fox_zone_crop.py` patterns there.

| Test | Evidence |
| --- | --- |
| `test_intent_synthetic.py` | Current/future leakage protection, group restrictions, medoid support, coordinate conversion, abstention, stable predictions after row shuffling |
| `test_zone_refinement.py` | Four-edge fitting, scaling, incomplete/false rectangles, camera registration, duplicate/wrong-frame rejection, selected-frame preservation |
| `test_frame_payload.py` | Frame geometry isolation, exact small-zone dimensions, target-audit queue and saved-review delivery |
| `test_fox_zone_crop.py` | Crop coordinate round trips, resolution handling, invalid predictions |
| `test_frame_overlay.js` | UI compilation, scoped manual overrides, scaling, invalid zones |

The JavaScript check also supports macOS JavaScriptCore. GitHub Actions runs the Python suite, synthetic demo, JavaScript overlay check, and syntax checks for all five interfaces.

Behavior tests do not measure real-world detector accuracy or inferred-intent accuracy. The private research workspace additionally contains `test_intent_pilot.py`, which checks actual reviewed merges, historical comparables, and leakage. That suite and its inputs are not part of the public checkout. Reproducing the empirical model comparison requires the reviewed records listed in [DATA.md](DATA.md).
