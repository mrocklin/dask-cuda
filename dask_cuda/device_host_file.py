from zict import Buffer, File, Func
from zict.common import ZictBase
from distributed.protocol import deserialize_bytes, serialize_bytes
from distributed.worker import weight

from functools import partial
import os


def _is_device_object(obj):
    """
    Check if obj is a device object, by checking if it has a
    __cuda_array_interface__ attributed
    """
    return hasattr(obj, "__cuda_array_interface__")


def _serialize_if_device(obj):
    """ Serialize an object if it's a device object """
    if _is_device_object(obj):
        return serialize_bytes(obj, on_error="raise")
    else:
        return obj


def _deserialize_if_device(obj):
    """ Deserialize an object if it's an instance of bytes """
    if isinstance(obj, bytes):
        return deserialize_bytes(obj)
    else:
        return obj


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

        self.device = dict()
        self.host = dict()
        self.disk = Func(
            partial(serialize_bytes, on_error="raise"), deserialize_bytes, File(path)
        )

        self.host_disk = Buffer(self.host, self.disk, memory_limit, weight=weight)
        self._device_host_func = Func(
            _serialize_if_device, _deserialize_if_device, self.host_disk
        )
        self.device_host_disk = Buffer(
            self.device, self._device_host_func, device_memory_limit, weight=weight
        )

    def __setitem__(self, key, value):
        if _is_device_object(value):
            self.device_host_disk[key] = value
        else:
            self.host_disk[key] = value

    def __getitem__(self, key):
        if key in self.host_disk:
            obj = self.host_disk[key]
            del self.host_disk[key]
            self.device_host_disk[key] = _deserialize_if_device(obj)

        if key in self.device_host_disk:
            return self.device_host_disk[key]
        else:
            raise KeyError

    def __len__(self):
        return len(self.device_host_disk)

    def __iter__(self):
        return iter(self.device_host_disk)

    def __delitem__(self, i):
        del self.device_host_disk[i]
