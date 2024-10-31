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
        raise NotImplementedError
        

    def read_async_num_stages(self, ids_device: torch.device):
        raise NotImplementedError
        if ids_device.type == "cuda":
            if self._tensor.is_cuda:
                # If the ids and the tensor are on cuda, no need for async.
                return 0
            return 1 if self.is_pinned() else 3
        else:
            return 1

    def size(self):
        return self._tensor.size()[1:]

    def count(self):
        return self._tensor.size()[0]

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
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


class WholeGraphMemoryFeature(Feature):
    r"""Adapted from TorchBasedFeature. A wrapper of pytorch based feature.
    """

    def __init__(self, wm_feature: "wgth.WholeMemoryEmbedding", metadata: Dict = None):
        super().__init__()
        import pylibwholegraph.torch as wgth
        assert isinstance(wm_feature, wgth.WholeMemoryEmbedding), (
            f"wm_feature in WholeGraphMemoryFeature must be torch.Tensor, "
            f"but got {type(wm_feature)}."
        )
        # assert dgl_feature.dim() > 1, (
        #     f"dimension of dgl_feature in TorchBasedFeature must be greater "
        #     f"than 1, but got {dgl_feature.dim()} dimension."
        # )
        # Make sure the tensor is contiguous.
        self._tensor = wm_feature# .contiguous()
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
        return self._tensor.gather(ids.cuda())

    def read_async(self, ids: torch.Tensor):
        raise NotImplementedError
        

    def read_async_num_stages(self, ids_device: torch.device):
        if ids_device.type == "cuda":
            if self._tensor.is_cuda:
                # If the ids and the tensor are on cuda, no need for async.
                return 0
            return 1 if self.is_pinned() else 3
        else:
            return 1

    def size(self):
        return self._tensor.size()[1:]

    def count(self):
        return self._tensor.size()[0]

    def update(self, value: torch.Tensor, ids: torch.Tensor = None):
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
