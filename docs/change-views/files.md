# Files changed

The optimizations span **11 source files** (9 added, 2 modified). This count excludes later documentation, launchers and evaluation archives. The implementation remains byte-identical to the measured retained commit.

| File | Area | Change |
|---|---|---|
| [nanochat/fp8.py](../../nanochat/fp8.py) | Model integration | Install default-off compiler/runtime hooks for the selected FP8 optimization preset. |
| [nanochat/gpt.py](../../nanochat/gpt.py) | Model integration | Enable the guarded native value-embedding gradient path. |
| [nanochat/scalar_scale_split.py](../../nanochat/scalar_scale_split.py) | FP8 scales | Compute repeated singleton scale arithmetic once in a small kernel, preserving typed arithmetic and stream-local lifetime. |
| [nanochat/persistent_scale_split.py](../../nanochat/persistent_scale_split.py) | FP8 scales | Extract singleton scale arithmetic from the matched persistent-reduction path. |
| [nanochat/norm_rowpack.py](../../nanochat/norm_rowpack.py) | Reductions | Pack four independent RMS backward rows per thread block, preserving the guarded reduction body. |
| [nanochat/max_rowpack.py](../../nanochat/max_rowpack.py) | Reductions | Pack eight independent row maxima per thread block, preserving FP32 and NaN-aware maximum semantics. |
| [nanochat/mlp_packed_layout.py](../../nanochat/mlp_packed_layout.py) | MLP layout | Transport BF16 operands in pairs while preserving BF16-to-FP32 conversion and arithmetic. |
| [nanochat/mlp_packed_tiles.py](../../nanochat/mlp_packed_tiles.py) | MLP launch | Search six guarded tile/warp configurations on the packed body; the retained full run uses 64×64 tiles and four warps. |
| [nanochat/lt_autotune.py](../../nanochat/lt_autotune.py) | Native GEMM | Build and load the native cuBLASLt selector with environment/source-bound caching. |
| [nanochat/lt_autotune.cpp](../../nanochat/lt_autotune.cpp) | Native GEMM | Select guarded backward FP8 GEMM algorithms while retaining precision and unsupported-path fallbacks. |
| [nanochat/native_value_embedding.py](../../nanochat/native_value_embedding.py) | Native gradients | Use native FP32 accumulation for value-embedding gradients; preserve existing producer/rounding boundaries. |

The [optimization preset](../../scripts/configs/faster_d24.sh) selects the measured paths. See the [exact upstream-to-retained patch](implementation.patch), [diffstat](diffstat.txt) and [cumulative checkpoint history](../optimization-history.md). A file's presence does not establish an independent speedup; no final remove-one-component ablation was run.

At Round28 the targeted MLP kernel's profiled duration decreased from about 91.96 to 68.27 ms/update. This attributes one kernel stage; it is not a whole-update gain estimate. Scale extraction gave the first large cumulative gain, followed by reduction, layout and native-path improvements.

Batch, data order, optimizer, schedule and model depth were preserved. The final N1 verdict allows documented FP32 optimizer-delta differences and validates bounded short trajectories; full-training BPB/CORE evaluation remains unrun. See [the H20 results](../h20-results.md).
