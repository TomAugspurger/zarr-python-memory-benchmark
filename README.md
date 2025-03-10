# zarr memory benchmark

This benchmark writes a `(100, 1000, 1000)` ndarray of `float32` data split into `10` chunks along the first dimension.

Component | Shape               | nbytes      |
--------- | ------------------- | ----------- |
Chunk     | `(10, 1000, 1000)`  | 40,000,000  |
Array     | `(100, 1000, 1000)` | 400,000,000 |

## Observations

### Read compressed

Peak memory usage is about 1.1 GiB.

Questions: what's the memory overhead of zstd? Do we know the uncompressed size? Can we tell zstd that?

Can we effectively `readinto` the decompression buffer? Maybe...

Why does `buf.as_numpy_array` apparently allocate memory?

### Read Uncompressed

For this special case, the peak memory usage ought to be the size of the ndarray. Currently, it's about 2x.

This is probably because `LocalStore` uses `path.read_bytes`, and then we put that into an array using `prototype.buffer.from_bytes`. See [here](https://github.com/zarr-developers/zarr-python/blob/38a241712b243ebb12a8f969e499789700a4334d/src/zarr/storage/_local.py#L29).

We would optimially use `readinto` into the memory backing the `out` ndarray. With enough effort that's probably doable. Given how rare uncompressed data is in practice, it might not be worthwhile.

## Profiles

- [read compressed](https://rawcdn.githack.com/TomAugspurger/zarr-python-memory-benchmark/refs/heads/main/reports/memray-flamegraph-read-compressed.html)
- [read uncompressed](https://rawcdn.githack.com/TomAugspurger/zarr-python-memory-benchmark/refs/heads/main/reports/memray-flamegraph-read-uncompressed.html)
