# Offline example

Run `python examples/synthetic_demo.py` from the repository root after installing `requirements-core.txt`.

`synthetic_inputs()` constructs one fictional query, seven reviewed historical comparables, and 30 historical profile pitches. `main()` calls the same `infer()` implementation used by the real pilot for all three variants and prints JSON. It makes no network requests and writes no files.

This fixture also supports the portable inference tests. It illustrates the input schema and software behavior; it is not a substitute for empirical evaluation.
