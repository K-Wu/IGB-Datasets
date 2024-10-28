"""DGL-DistTensor-based feature store for GraphBolt. Adapted from https://github.com/dmlc/dgl/blob/master/python/dgl/graphbolt/impl/torch_based_feature_store.py"""
from typing import Dict, List
import textwrap

import numpy as np
import torch

from dgl.graphbolt.feature_store import Feature
from dgl.graphbolt.impl.basic_feature_store import BasicFeatureStore
from dgl.graphbolt.impl.ondisk_metadata import OnDiskFeatureData

from dgl.distributed.dist_tensor import DistTensor


class TorchDistTensorFeature(Feature):
    r"""Adapted from TorchBasedFeature. A wrapper of pytorch based feature.

    Initialize a torch based feature store by a torch feature.
    Note that the feature can be either in memory or on disk.

    Parameters
    ----------
    torch_feature : torch.Tensor
        The torch feature.
        Note that the dimension of the tensor should be greater than 1.

    Examples
    --------
    >>> import torch
    >>> from dgl import graphbolt as gb

    1. The feature is in memory.

    >>> torch_feat = torch.arange(10).reshape(2, -1)
    >>> feature = gb.TorchBasedFeature(torch_feat)
    >>> feature.read()
    tensor([[0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]])
    >>> feature.read(torch.tensor([0]))
    tensor([[0, 1, 2, 3, 4]])
    >>> feature.update(torch.tensor([[1 for _ in range(5)]]),
    ...                      torch.tensor([1]))
    >>> feature.read(torch.tensor([0, 1]))
    tensor([[0, 1, 2, 3, 4],
            [1, 1, 1, 1, 1]])
    >>> feature.size()
    torch.Size([5])

    2. The feature is on disk. Note that you can use gb.numpy_save_aligned as a
    replacement for np.save to potentially get increased performance.

    >>> import numpy as np
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> np.save("/tmp/arr.npy", arr)
    >>> torch_feat = torch.from_numpy(np.load("/tmp/arr.npy", mmap_mode="r+"))
    >>> feature = gb.TorchBasedFeature(torch_feat)
    >>> feature.read()
    tensor([[1, 2],
            [3, 4]])
    >>> feature.read(torch.tensor([0]))
    tensor([[1, 2]])

    3. Pinned CPU feature.

    >>> torch_feat = torch.arange(10).reshape(2, -1).pin_memory()
    >>> feature = gb.TorchBasedFeature(torch_feat)
    >>> feature.read().device
    device(type='cuda', index=0)
    >>> feature.read(torch.tensor([0]).cuda()).device
    device(type='cuda', index=0)
    """

    def __init__(self, dgl_feature: DistTensor, metadata: Dict = None):
        super().__init__()
        assert isinstance(dgl_feature, DistTensor), (
            f"dgl_feature in TorchBasedFeature must be torch.Tensor, "
            f"but got {type(dgl_feature)}."
        )
        # assert dgl_feature.dim() > 1, (
        #     f"dimension of dgl_feature in TorchBasedFeature must be greater "
        #     f"than 1, but got {dgl_feature.dim()} dimension."
        # )
        # Make sure the tensor is contiguous.
        self._tensor = dgl_feature# .contiguous()
        self._metadata = metadata

    def __del__(self):
        # torch.Tensor.pin_memory() is not an inplace operation. To make it
        # truly in-place, we need to use cudaHostRegister. Then, we need to use
        # cudaHostUnregister to unpin the tensor in the destructor.
        # https://github.com/pytorch/pytorch/issues/32167#issuecomment-753551842
        return

    def read(self, ids: torch.Tensor = None):
        """Read the feature by index.

        If the feature is on pinned CPU memory and `ids` is on GPU or pinned CPU
        memory, it will be read by GPU and the returned tensor will be on GPU.
        Otherwise, the returned tensor will be on CPU.

        Parameters
        ----------
        ids : torch.Tensor, optional
            The index of the feature. If specified, only the specified indices
            of the feature are read. If None, the entire feature is returned.

        Returns
        -------
        torch.Tensor
            The read feature.
        """
        if ids is None:
            return self._tensor
        return self._tensor[ids.cpu()]

    def read_async(self, ids: torch.Tensor):
        r"""Read the feature by index asynchronously.

        Parameters
        ----------
        ids : torch.Tensor
            The index of the feature. Only the specified indices of the
            feature are read.
        Returns
        -------
        A generator object.
            The returned generator object returns a future on
            ``read_async_num_stages(ids.device)``\ th invocation. The return result
            can be accessed by calling ``.wait()``. on the returned future object.
            It is undefined behavior to call ``.wait()`` more than once.

        Examples
        --------
        >>> import dgl.graphbolt as gb
        >>> feature = gb.Feature(...)
        >>> ids = torch.tensor([0, 2])
        >>> for stage, future in enumerate(feature.read_async(ids)):
        ...     pass
        >>> assert stage + 1 == feature.read_async_num_stages(ids.device)
        >>> result = future.wait()  # result contains the read values.
        """
        raise NotImplementedError
        assert self._tensor.device.type == "cpu"
        if ids.is_cuda and self.is_pinned():
            current_stream = torch.cuda.current_stream()
            host_to_device_stream = get_host_to_device_uva_stream()
            host_to_device_stream.wait_stream(current_stream)
            with torch.cuda.stream(host_to_device_stream):
                ids.record_stream(torch.cuda.current_stream())
                values = index_select(self._tensor, ids)
                values.record_stream(current_stream)
                values_copy_event = torch.cuda.Event()
                values_copy_event.record()

            yield _Waiter(values_copy_event, values)
        elif ids.is_cuda:
            ids_device = ids.device
            current_stream = torch.cuda.current_stream()
            device_to_host_stream = get_device_to_host_uva_stream()
            device_to_host_stream.wait_stream(current_stream)
            with torch.cuda.stream(device_to_host_stream):
                ids.record_stream(torch.cuda.current_stream())
                ids = ids.to(self._tensor.device, non_blocking=True)
                ids_copy_event = torch.cuda.Event()
                ids_copy_event.record()

            yield  # first stage is done.

            ids_copy_event.synchronize()
            values = torch.ops.graphbolt.index_select_async(self._tensor, ids)
            yield

            host_to_device_stream = get_host_to_device_uva_stream()
            with torch.cuda.stream(host_to_device_stream):
                values_cuda = values.wait().to(ids_device, non_blocking=True)
                values_cuda.record_stream(current_stream)
                values_copy_event = torch.cuda.Event()
                values_copy_event.record()

            yield _Waiter(values_copy_event, values_cuda)
        else:
            yield torch.ops.graphbolt.index_select_async(self._tensor, ids)

    def read_async_num_stages(self, ids_device: torch.device):
        """The number of stages of the read_async operation. See read_async
        function for directions on its use. This function is required to return
        the number of yield operations when read_async is used with a tensor
        residing on ids_device.

        Parameters
        ----------
        ids_device : torch.device
            The device of the ids parameter passed into read_async.
        Returns
        -------
        int
            The number of stages of the read_async operation.
        """
        if ids_device.type == "cuda":
            if self._tensor.is_cuda:
                # If the ids and the tensor are on cuda, no need for async.
                return 0
            return 1 if self.is_pinned() else 3
        else:
            return 1

    def size(self):
        """Get the size of the feature.

        Returns
        -------
        torch.Size
            The size of the feature.
        """
        return self._tensor.size()[1:]

    def count(self):
        """Get the count of the feature.

        Returns
        -------
        int
            The count of the feature.
        """
        return self._tensor.size()[0]

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
        """Update the feature store.

        Parameters
        ----------
        value : torch.Tensor
            The updated value of the feature.
        ids : torch.Tensor, optional
            The indices of the feature to update. If specified, only the
            specified indices of the feature will be updated. For the feature,
            the `ids[i]` row is updated to `value[i]`. So the indices and value
            must have the same length. If None, the entire feature will be
            updated.
        """
        raise NotImplementedError

    def metadata(self):
        """Get the metadata of the feature.

        Returns
        -------
        Dict
            The metadata of the feature.
        """
        return (
            self._metadata if self._metadata is not None else super().metadata()
        )

    def pin_memory_(self):
        """In-place operation to copy the feature to pinned memory. Returns the
        same object modified in-place."""
        # torch.Tensor.pin_memory() is not an inplace operation. To make it
        # truly in-place, we need to use cudaHostRegister. Then, we need to use
        # cudaHostUnregister to unpin the tensor in the destructor.
        # https://github.com/pytorch/pytorch/issues/32167#issuecomment-753551842
        
        raise NotImplementedError

    def is_pinned(self):
        """Returns True if the stored feature is pinned."""
        raise NotImplementedError

    def to(self, device):  # pylint: disable=invalid-name
        """Copy `TorchBasedFeature` to the specified device."""
        # copy.copy is a shallow copy so it does not copy tensor memory.
        raise NotImplementedError

    def __repr__(self) -> str:
        ret = (
            "{Classname}(\n"
            "    feature={feature},\n"
            "    metadata={metadata},\n"
            ")"
        )

        feature_str = textwrap.indent(
            str(self._tensor), " " * len("    feature=")
        ).strip()
        metadata_str = textwrap.indent(
            str(self.metadata()), " " * len("    metadata=")
        ).strip()

        return ret.format(
            Classname=self.__class__.__name__,
            feature=feature_str,
            metadata=metadata_str,
        )
