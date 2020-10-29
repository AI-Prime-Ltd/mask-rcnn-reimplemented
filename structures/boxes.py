# -*- coding: utf-8 -*-
# ref: https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/boxes.py

import math
import logging
import numpy as np
from enum import IntEnum, unique
from typing import Any, List, Tuple, Union
import torch


logger = logging.getLogger(__name__)
_RawBoxType = Union[List[float], Tuple[float, ...], torch.Tensor, np.ndarray]


@unique
class BoxMode(IntEnum):
    """
    Enum of different ways to represent a box.
    """

    XYXY_ABS = 0
    """
    (x0, y0, x1, y1) in absolute floating points coordinates.
    The coordinates in range [0, width or height].
    """
    XYWH_ABS = 1
    """
    (x0, y0, w, h) in absolute floating points coordinates.
    """
    XYXY_REL = 2
    """
    Not yet supported!
    (x0, y0, x1, y1) in range [0, 1]. They are relative to the size of the image.
    """
    XYWH_REL = 3
    """
    Not yet supported!
    (x0, y0, w, h) in range [0, 1]. They are relative to the size of the image.
    """
    XYWHA_ABS = 4
    """
    (xc, yc, w, h, a) in absolute floating points coordinates.
    (xc, yc) is the center of the rotated box, and the angle a is in degrees ccw.
    """


class Boxes:
    """
    This structure stores a list of axis-aligned boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)
    Attributes:
        tensor (torch.Tensor): float matrix of Nx4 or Nx5.
        mode (BoxMode): the mode to represent each box.

    Note:
        Currently, rotated boxes are internally transformed into axis-aligned boxes,
        this may change the size of your rotated boxes.
    """

    def __init__(self, raw_boxes: _RawBoxType, mode=BoxMode.XYXY_ABS):
        """
        Args:
            raw_boxes (_RawBoxType): a Nx4 or Nx5 matrix.  Each row is a box.
            mode (BoxMode): the mode to represent each box, the default is `BoxMode.XYXY_ABS`.
        """
        device = raw_boxes.device if isinstance(raw_boxes, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(raw_boxes, dtype=torch.float32, device=device)

        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((0, 4)).to(dtype=torch.float32, device=device)

        tensor = convert(tensor, mode, BoxMode.XYXY_ABS)
        assert tensor.dim() == 2 and tensor.size(-1) == 4, tensor.size()

        self._tensor = tensor
        self._mode = mode

    @property
    def tensor(self):
        return convert(self._tensor, BoxMode.XYXY_ABS, self._mode)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        self._mode = new_mode

    def clone(self) -> "Boxes":
        """
        Clone the Boxes.
        Returns:
            Boxes
        """
        return Boxes(self._tensor.clone(), mode=self._mode)

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any):
        if "mode" in kwargs.keys():
            assert isinstance(kwargs["mode"], BoxMode)
            self._mode = kwargs.pop("mode")
        return Boxes(self._tensor.to(*args, **kwargs))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.
        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self._tensor
        area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
        return area

    def clip_(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].
        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self._tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        self._tensor[:, 0].clamp_(min=0, max=w)
        self._tensor[:, 1].clamp_(min=0, max=h)
        self._tensor[:, 2].clamp_(min=0, max=w)
        self._tensor[:, 3].clamp_(min=0, max=h)

    def clip(self, box_size: Tuple[int, int]) -> "Boxes":
        """
        Clip the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].
        Args:
            box_size (height, width): The clipping box's size.
        Returns:
            torch.Tensor: clipped boxes.
        """
        clone = self.clone()
        clone.clip_(box_size)
        return clone

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.
        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self._tensor
        widths = box[:, 2] - box[:, 0]
        heights = box[:, 3] - box[:, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item):
        """
        Args:
            item: int, slice, or a BoolTensor
        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.
        The following usage are allowed:
        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.
        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self._tensor[item].view(1, -1))
        b = self._tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        return self._tensor.shape[0]

    def __repr__(self) -> str:
        return "Boxes(" + str(self._tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".
        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self._tensor[..., 0] >= -boundary_threshold)
            & (self._tensor[..., 1] >= -boundary_threshold)
            & (self._tensor[..., 2] < width + boundary_threshold)
            & (self._tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self._tensor[:, :2] + self._tensor[:, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self._tensor[:, 0::2] *= scale_x
        self._tensor[:, 1::2] *= scale_y

    # noinspection PyMethodFirstArgAssignment
    @classmethod
    def cat(cls, boxes_list: List["Boxes"]) -> "Boxes":
        """
        Concatenates a list of Boxes into a single Boxes
        Arguments:
            boxes_list (list[Boxes])
        Returns:
            Boxes: the concatenated Boxes
        """
        if torch.jit.is_scripting():
            # https://github.com/pytorch/pytorch/issues/18627
            # 1. staticmethod can be used in torchscript, But we can not use
            # `type(boxes).staticmethod` because torchscript only supports function
            # `type` with input type `torch.Tensor`.
            # 2. classmethod is not fully supported by torchscript. We explicitly assign
            # cls to Box as a workaround to get torchscript support.
            cls = Boxes
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b._tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (4,) at a time.
        """
        yield from convert(self._tensor, BoxMode.XYXY_ABS, self._mode)


def convert(boxes: _RawBoxType, from_mode: BoxMode, to_mode: BoxMode) -> _RawBoxType:
    """
    Args:
        boxes: can be a k-tuple, k-list or an Nxk array/tensor, where k = 4 or 5
        from_mode: input BoxMode
        to_mode: output BoxMode
        shape: used when input BoxMode is relative
    Returns:
        The converted box of the same type.
    """

    if from_mode == to_mode:
        return boxes

    original_type = type(boxes)
    is_numpy = isinstance(boxes, np.ndarray)
    single_box = isinstance(boxes, (list, tuple))
    if single_box:
        assert len(boxes) == 4 or len(boxes) == 5, (
            "BoxMode.convert takes either a k-tuple/list or an Nxk array/tensor,"
            " where k == 4 or 5"
        )
        arr = torch.tensor(boxes)[None, :]
    else:
        # avoid modifying the input box
        if is_numpy:
            arr = torch.from_numpy(np.asarray(boxes)).clone()
        else:
            arr = boxes.clone()

    assert to_mode.value not in [
        BoxMode.XYXY_REL,
        BoxMode.XYWH_REL,
    ] and from_mode.value not in [
        BoxMode.XYXY_REL,
        BoxMode.XYWH_REL,
    ], "Relative mode not yet supported!"

    if from_mode == BoxMode.XYWHA_ABS and to_mode == BoxMode.XYXY_ABS:
        assert (
            arr.shape[-1] == 5
        ), "The last dimension of input shape must be 5 for XYWHA format"
        original_dtype = arr.dtype
        arr = arr.double()

        w = arr[:, 2]
        h = arr[:, 3]
        a = arr[:, 4]
        c = torch.abs(torch.cos(a * math.pi / 180.0))
        s = torch.abs(torch.sin(a * math.pi / 180.0))
        # This basically computes the horizontal bounding rectangle of the rotated box
        new_w = c * w + s * h
        new_h = c * h + s * w

        # convert center to top-left corner
        arr[:, 0] -= new_w / 2.0
        arr[:, 1] -= new_h / 2.0
        # bottom-right corner
        arr[:, 2] = arr[:, 0] + new_w
        arr[:, 3] = arr[:, 1] + new_h

        arr = arr[:, :4].to(dtype=original_dtype)
    elif from_mode == BoxMode.XYWH_ABS and to_mode == BoxMode.XYWHA_ABS:
        original_dtype = arr.dtype
        arr = arr.double()
        arr[:, 0] += arr[:, 2] / 2.0
        arr[:, 1] += arr[:, 3] / 2.0
        angles = torch.zeros((arr.shape[0], 1), dtype=arr.dtype)
        arr = torch.cat((arr, angles), axis=1).to(dtype=original_dtype)
    else:
        if to_mode == BoxMode.XYXY_ABS and from_mode == BoxMode.XYWH_ABS:
            arr[:, 2] += arr[:, 0]
            arr[:, 3] += arr[:, 1]
        elif from_mode == BoxMode.XYXY_ABS and to_mode == BoxMode.XYWH_ABS:
            arr[:, 2] -= arr[:, 0]
            arr[:, 3] -= arr[:, 1]
        else:
            raise NotImplementedError(
                "Conversion from BoxMode {} to {} is not supported yet".format(
                    from_mode, to_mode
                )
            )

    if single_box:
        return original_type(arr.flatten().tolist())
    if is_numpy:
        return arr.numpy()
    else:
        return arr


def pairwise_intersection(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the intersection area between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax)
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: intersection, sized [N,M].
    """
    boxes1, boxes2 = boxes1._tensor, boxes2._tensor
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def pairwise_ioa(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Similar to pariwise_iou but compute the IoA (intersection over boxes2 area).
    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.
    Returns:
        Tensor: IoA, sized [N,M].
    """
    area2 = boxes2.area()  # [M]
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    ioa = torch.where(
        inter > 0, inter / area2, torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return ioa


def matched_boxlist_iou(boxes1: Boxes, boxes2: Boxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes. The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix
    Args:
        boxes1: (Boxes) bounding boxes, sized [N,4].
        boxes2: (Boxes) bounding boxes, sized [N,4].
    Returns:
        Tensor: iou, sized [N].
    """
    assert len(boxes1) == len(
        boxes2
    ), "boxlists should have the same" "number of entries, got {}, {}".format(
        len(boxes1), len(boxes2)
    )
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1._tensor, boxes2._tensor
    lt = torch.max(box1[:, :2], box2[:, :2])  # [N,2]
    rb = torch.min(box1[:, 2:], box2[:, 2:])  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou
