import zarr
import zarr.abc.store
import numpy as np
import zarr.storage
import memray
# these are for GPU. Split this outt
# import kvikio_zarr_v3
# import nvtx
# import cupy as cp

SHAPE = (100, 1000, 1000)
CHUNKS = (10, 1000, 1000)

# Questions:
# - What's the speed of light of light from disk to GPU?
#   - Try this from Python and a simple C program.
# - What's the speed of light from GPU to ndarray (uncompressed)?
#   - This should just be a view / pointer thing
# - What's the speed of light from GPU to ndarray (compressed)?
#   - this is the throughput of the decompression hardware / software

UNCOMPRESSED_PATH = "/uncompressed"
COMPRESSED_PATH = "/compressed"
UNCOMPRESSED_CONFIG = {"numeric": [], "string": [], "bytes": []}



def read_compressed(store: zarr.abc.store.Store, path: str) -> None:
    with memray.Tracker(f"read-{path.lower().lstrip("/")}.bin", native_traces=True):
        z = zarr.open_array(store, mode="r", path=path)
        return z[:]


def write_compressed(store: zarr.abc.store.Store, arr: np.ndarray) -> None:
    with memray.Tracker("write-compressed.bin", native_traces=True):
        z = zarr.create_array(store, name=COMPRESSED_PATH, shape=arr.shape, dtype=arr.dtype, overwrite=True)
        z[:] = arr


def write_uncompressed(store: zarr.abc.store.Store, arr: np.ndarray) -> None:
    with zarr.config.set({"array.v3_default_compressors": UNCOMPRESSED_CONFIG}):
        with memray.Tracker("write-uncompressed.bin", native_traces=True):
            z = zarr.create_array(store, name=UNCOMPRESSED_PATH, shape=arr.shape, dtype=arr.dtype, overwrite=True)
            z[:] = arr


def main():
    arr_cpu = np.random.randn(*SHAPE).astype("float32")
    # arr_gpu = cp.asarray(arr_cpu)

    store = zarr.storage.LocalStore("/tmp/data.zarr")

    print("CPU - write - uncompressed")
    write_uncompressed(store, arr_cpu)
    print("CPU - write - compressed")
    write_compressed(store, arr_cpu)

    print("CPU - read - uncompressed")
    read_compressed(store, path=UNCOMPRESSED_PATH)

    print("CPU - read - compressed")
    read_compressed(store, path=COMPRESSED_PATH)

if __name__ == "__main__":
    main()
