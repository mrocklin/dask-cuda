from zict import Buffer, File, Func
from zict.common import ZictBase
from distributed.protocol import (
    deserialize,
    deserialize_bytes,
    serialize,
    serialize_bytelist,
    register_generic,
)
from dask.sizeof import sizeof
from distributed.utils import nbytes
from distributed.worker import weight

from numba import cuda
import os

from .is_device_object import is_device_object


# Register sizeof for Numba DeviceNDArray while Dask doesn't add it
if not hasattr(sizeof, "register_numba"):

    @sizeof.register_lazy("numba")
    def register_numba():
        import numba

        @sizeof.register(numba.cuda.cudadrv.devicearray.DeviceNDArray)
        def sizeof_numba_devicearray(x):
            return int(x.nbytes)


class DeviceSerialized:
    """ Store device object on the host

    This stores a device-side object as

    1.  A msgpack encodable header
    2.  A list of objects that are returned by calling
        `numba.cuda.as_cuda_array(f).copy_to_host()`
        which are typically NumPy arrays
    """

    def __init__(self, header, parts):
        self.header = header
        self.parts = parts

    def __sizeof__(self):
        return sum(map(nbytes, self.parts))


register_generic(DeviceSerialized)


def device_to_host(obj: object) -> DeviceSerialized:
    header, frames = serialize(obj, serializers=["cuda"])
    is_cuda = [hasattr(f, "__cuda_array_interface__") for f in frames]
    frames = [
        cuda.as_cuda_array(f).copy_to_host() if ic else f
        for ic, f in zip(is_cuda, frames)
    ]
    header = {"sub-header": header, "is-cuda": is_cuda}
    return DeviceSerialized(header, frames)


def host_to_device(s: DeviceSerialized) -> object:
    is_cuda = s.header["is-cuda"]
    header = s.header["sub-header"]
    frames = [cuda.to_device(f) if ic else f for ic, f in zip(is_cuda, s.parts)]
    return deserialize(header, frames)


class DeviceHostFile(ZictBase):
    """ Manages serialization/deserialization of objects.

    Three LRU cache levels are controlled, for device, host and disk.
    Each level takes care of serializing objects once its limit has been
    reached and pass it to the subsequent level. Similarly, each cache
    may deserialize the object, but storing it back in the appropriate
    cache, depending on the type of object being deserialized.

    Parameters
    ----------
    device_memory_limit: int
        Number of bytes of CUDA device memory for device LRU cache,
        spills to host cache once filled.
    memory_limit: int
        Number of bytes of host memory for host LRU cache, spills to
        disk once filled.
    local_dir: path
        Path where to store serialized objects on disk
    """

    def __init__(
        self, device_memory_limit=None, memory_limit=None, local_dir="dask-worker-space"
    ):
        path = os.path.join(local_dir, "storage")

        self.host_func = dict()
        self.disk_func = Func(serialize_bytelist, deserialize_bytes, File(path))
        self.host_buffer = Buffer(
            self.host_func, self.disk_func, memory_limit, weight=weight
        )

        self.device_keys = set()
        self.device_func = dict()
        self.device_host_func = Func(device_to_host, host_to_device, self.host_buffer)
        self.device_buffer = Buffer(
            self.device_func, self.device_host_func, device_memory_limit, weight=weight
        )

        self.device = self.device_buffer.fast.d
        self.host = self.host_buffer.fast.d
        self.disk = self.host_buffer.slow.d

        # For Worker compatibility only, where `fast` is host memory buffer
        self.fast = self.host_buffer.fast

    def __setitem__(self, key, value):
        if is_device_object(value):
            self.device_keys.add(key)
            self.device_buffer[key] = value
        else:
            self.host_buffer[key] = value

    def __getitem__(self, key):
        if key in self.device_keys:
            return self.device_buffer[key]
        elif key in self.host_buffer:
            return self.host_buffer[key]
        else:
            raise KeyError(key)

    def __len__(self):
        return len(self.device_buffer)

    def __iter__(self):
        return iter(self.device_buffer)

    def __delitem__(self, key):
        self.device_keys.discard(key)
        del self.device_buffer[key]
