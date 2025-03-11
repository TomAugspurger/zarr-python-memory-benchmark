"""
What's the speed of light, in terms of memory usage, for reading this into
memory?

**Uncompressed data**

For uncomprssed arrays, we need just a single allocation of the output array.
All reads can be done directly into (slices of) that output array. Probably... I
see some comments in the codec pipeline about stuff that might break that but
let's ignore those for now.

**Compressed data**

Compressed data is slightly harder, but not by much. We can use the file
metadata to pre-allocate the intermediate buffers (one per file), and
read the compressed data into those buffers.

`numcodecs.zstd.Zstd` is able to decode into an `out` array, so once
we just need to pass the (in-memoyr) compressed buffers into the `out`
just like before.
"""

import os
import pathlib
import numpy as np
import memray
import numcodecs
import zarr.storage
import zarr


SHAPE = (100, 1000, 1000)
CHUNKS = (10, 1000, 1000)


def read_file(file: pathlib.Path, out: np.ndarray) -> None:
    with open(file, "rb") as f:
        f.readinto(out)


def read_uncompressed():
    output = pathlib.Path("reports/sol-read-uncompressed.bin")
    output.unlink(missing_ok=True)

    root = pathlib.Path("/tmp/data.zarr/uncompressed/c/")
    paths = sorted(
        [p for p in root.glob("**/*") if p.is_file()],
        key=lambda p: int(p.parent.parent.stem),
    )
    stride = 10
    slices = [slice(i * stride, (i + 1) * stride) for i in range(len(paths))]

    with memray.Tracker(output, native_traces=True):
        out = np.empty(SHAPE, dtype="float32")
        for path, slice_ in zip(paths, slices):
            read_file(path, out[slice_])

    store = zarr.storage.LocalStore("/tmp/data.zarr")
    expected = zarr.open_array(store, mode="r", path="uncompressed/")
    np.testing.assert_array_equal(out, expected[:])


def read_compressed():
    output = pathlib.Path("reports/sol-read-compressed.bin")
    output.unlink(missing_ok=True)

    codec = numcodecs.Zstd(level=0)

    root = pathlib.Path("/tmp/data.zarr/compressed/c/")
    paths = sorted(
        [p for p in root.glob("**/*") if p.is_file()],
        key=lambda p: int(p.parent.parent.stem),
    )
    stride = 10
    slices = [slice(i * stride, (i + 1) * stride) for i in range(len(paths))]

    with memray.Tracker(output, native_traces=True):
        out = np.empty(SHAPE, dtype="float32")
        tmp_buffers = [np.empty(os.path.getsize(path), dtype="b") for path in paths]

        for path, slice_, tmp in zip(paths, slices, tmp_buffers):
            read_file(path, tmp)
            codec.decode(tmp, out[slice_])

    store = zarr.storage.LocalStore("/tmp/data.zarr")
    expected = zarr.open_array(store, mode="r", path="compressed")
    np.testing.assert_array_equal(out, expected[:])


if __name__ == "__main__":
    read_uncompressed()
    read_compressed()
