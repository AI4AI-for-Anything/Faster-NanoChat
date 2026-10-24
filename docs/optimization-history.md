# Optimization history

Fifty numbered exploration rounds produced eight retained implementation commits. Values below are cumulative original-baseline full-check speedups, not independent ablations. Separately timed checkpoint differences also contain measurement variation.

| Round | Retained change | Cumulative speedup |
|---|---|---:|
| R11 | FP8 scalar-scale extraction | 1.0364756× |
| R15 | RMS backward row packing | 1.0406241× |
| R16 | Row-maximum packing | 1.0522507× |
| R18 | Native backward FP8 GEMM selection | 1.0559014× |
| R22 | Persistent reduction scale extraction | 1.0652243× |
| R28 | MLP operand packing | 1.0699262× |
| R33 | Native value-embedding gradient accumulation | 1.0711360× |
| R43 | Packed MLP tile selection | 1.0739769× |

Final unchanged-source validation: **1.074368586×**. The 1.5× target was not reached. The implementation saved in Round43 remains the retained version. Round44–50 did not establish additional benefit; Round50's full PASS alone did not justify promotion.

The compact [chart data](performance-views/history.json) records the measured timing windows, checkpoint scores, original/retained commits and final report SHA256. The [renderer](performance-views/render.py) regenerates all three figures without GPU work:

```bash
python docs/performance-views/render.py
```

Matplotlib is available through the upstream development dependency group (`uv sync --extra gpu --group dev`). Full experimental logs, frozen evaluators, duplicate source snapshots and numerical tensor fixtures are preserved in the separate research archive, outside this source repository. See [the H20 results](h20-results.md) for validation scope.
