"""Run the real inference function on invented pitches; no downloads or API keys."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_intent_pilot import VARIANTS, infer


def synthetic_inputs():
    """Small deterministic fixture, intentionally unrelated to any real player."""
    query = SimpleNamespace(
        pitcher=1, pitch_type="CH", stand="R", game_date="2025-06-10",
        balls=1, strikes=2, model_target_x_01=0.65, model_target_y_01=0.25,
        sz_bot=1.5, sz_top=3.5,
    )
    audited = pd.DataFrame([
        dict(pitch_uid=f"synthetic_{i}", pitcher=1, pitch_type="CH", stand="R",
             game_date=f"2025-05-{i + 1:02d}", balls=1, strikes=2,
             model_target_x_01=0.65 + (i - 3) * 0.015,
             model_target_y_01=0.25 + (i % 3 - 1) * 0.015,
             model_miss_x_01=dx, model_miss_y_01=dy,
             pfx_x=-1.0 + i * 0.015, pfx_z=0.4 + i * 0.01,
             release_speed=86.0 + i * 0.1, release_spin_rate=1750 + i * 10,
             description="swinging_strike" if i % 2 else "ball",
             estimated_woba_using_speedangle=float("nan"))
        for i, (dx, dy) in enumerate([
            (0.08, -0.14), (0.10, -0.19), (0.12, -0.21), (0.09, -0.18),
            (0.14, -0.22), (0.11, -0.17), (0.07, -0.16),
        ])
    ])
    history = pd.DataFrame([
        dict(pitcher=1, pitch_type="CH", game_date="2025-05-01",
             at_bat_number=i + 1, pitch_number=1, pfx_x=-0.95, pfx_z=0.43,
             release_speed=86.3, release_spin_rate=1780)
        for i in range(30)
    ])
    return query, audited, history


def main():
    query, audited, history = synthetic_inputs()
    result = {
        "data": "SYNTHETIC: demonstrates software behavior, not model accuracy",
        "glove_target": {"x_01": query.model_target_x_01, "y_01": query.model_target_y_01},
        "variants": {
            variant: infer(query, audited, history,
                           outcome_weight=variant != "no_outcome_weight",
                           shape_weight=variant != "no_shape_weight")
            for variant in VARIANTS
        },
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
