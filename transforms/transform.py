import inspect
import pprint
from abc import ABCMeta, abstractmethod
from functools import partial
from itertools import chain
from typing import Any, Callable, List, Optional, TypeVar, Tuple, Dict, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .functional import adjust_color, adjust_contrast, adjust_brightness
from .transform_utils import to_numpy, to_float_tensor


class Transform(metaclass=ABCMeta):
    """
    Base class for implementations of **deterministic** transformations for
    image and other data structures. "Deterministic" requires that the output
    of all methods of this class are deterministic w.r.t their input arguments.
    Note that this is different from (random) data augmentations. To perform
    data augmentations in training, there should be a higher-level policy that
    generates these transform ops.
    Each transform op may handle several data types, e.g.: image, coordinates,
    segmentation, bounding boxes, with its ``apply_*`` methods. Some of
    them have a default implementation, but can be overwritten if the default
    isn't appropriate. See documentation of each pre-defined ``apply_*`` methods
    for details. Note that The implementation of these method may choose to
    modify its input data in-place for efficient transformation.
    The class can be extended to support arbitrary new data types with its
    :meth:`register_type` method.
    """

    def _set_attributes(self, params: Optional[List[Any]] = None) -> None:
        """
        Set attributes from the input list of parameters.
        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def apply(self, *imgs: np.ndarray, **kwargs: Union[Tuple[np.ndarray, ...], Dict[str, np.ndarray]]):
        """
        Apply transforms at the same time on different types of data.
        Args:
            imgs (np.ndarray): images to perform transformation.
            kwargs (datatype -> tuple or dict of ndarrays): data to perform transformation.
        Returns:
            tuple of images if only imgs are provided else dict of tuples or a single dict.
        Examples:
            ```
            >>> # simple usage: apply transform to images only
            >>> img1, img2 = transform.apply(img1, img2)
            >>>
            >>> # apply transform to two images (image batches) and coordinates
            >>> transformed = transform.apply(image=(img1, img2), coords=(kpts, ))
            >>> img1 = transformed["image"][0]
            >>>
            >>> # transform images named 'img1', 'img2' and coordinates named 'kpts'.
            >>> transformed = transform.apply(
            >>>     image={"img1": img1, "img2": img2},
            >>>     coords={"kpts": kpts}
            >>> )
            >>> img1 = transformed["img1"]
            ```
        """
        assert not (imgs and kwargs), "please provide either imgs or kwargs but not both."
        if imgs:  # simple case
            return tuple(self.apply_image(img) for img in imgs)
        else:  # kwargs case
            for dt in kwargs.keys():  # check unique
                assert hasattr(self, "apply_" + dt), f"{self.__class__} for type {dt} is undefined."
            if all(isinstance(v, tuple) for v in kwargs.values()):
                res = {}
                img_data = kwargs.pop("image", None)
                if img_data:
                    res["image"] = tuple(self.apply_image(data) for data in img_data)
                coords_data = kwargs.pop("coords", None)
                if coords_data:
                    res["coords"] = tuple(self.apply_coords(data) for data in coords_data)
                res.update(
                    {dt: tuple(getattr(self, "apply_" + dt)(data) for data in kwargs[dt]) for dt in kwargs.keys()})
                return res
            elif all(isinstance(v, dict) for v in kwargs.values()):
                assert len(set(chain(*kwargs.values()))) == sum(len(v.keys()) for v in kwargs.values()), \
                    "keys must be unique for all data types"
                res = {}
                img_data = kwargs.pop("image", None)
                if img_data:
                    res["image"] = {k: self.apply_image(v) for k, v in img_data.items()}
                coords_data = kwargs.pop("coords", None)
                if coords_data:
                    res["coords"] = {k: self.apply_coords(v) for k, v in coords_data.items()}
                for dt in kwargs.keys():
                    res.update({k: getattr(self, "apply_" + dt)(v) for k, v in kwargs[dt].items()})
                return res
            else:
                raise ValueError("each value in kwargs should be either a dict or a tuple but not mixed.")

    @abstractmethod
    def apply_image(self, img: np.ndarray):
        """
        Apply the transform on an image.
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: image after apply the transformation.

        Note:
            By default, apply_segmentation directly use this function.
        """

    def apply_coords(self, coords: np.ndarray):
        """
        Apply the transform on coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is (x, y).
        Returns:
            ndarray: coordinates after apply the transformation.
        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].
            This function should correctly transform coordinates outside the image as well.
            By default, apply_boxes and apply_polygons are implemented with this function.
        """

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply the transform on a full-image segmentation.
        By default will just perform "apply_image".
        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
            or bool dtype.
        Returns:
            ndarray: segmentation after apply the transformation.
        """
        return self.apply_image(segmentation)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        """
        Apply the transform on axis-aligned boxes. By default will transform
        the corner points and use their minimum/maximum to create a new
        axis-aligned box. Note that this default may change the size of your
        box, e.g. after rotations.
        Args:
            box (ndarray): Nx4 floating point array of XYXY format in absolute
                coordinates.
        Returns:
            ndarray: box after apply the transformation.
        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].
            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        # Indexes of converting (x0, y0, x1, y1) box into 4 coordinates of
        # ([x0, y0], [x1, y0], [x0, y1], [x1, y1]).
        idxs = np.array([(0, 1), (2, 1), (0, 3), (2, 3)]).flatten()
        coords = np.asarray(box).reshape(-1, 4)[:, idxs].reshape(-1, 2)
        coords = self.apply_coords(coords).reshape((-1, 4, 2))
        minxy = coords.min(axis=1)
        maxxy = coords.max(axis=1)
        trans_boxes = np.concatenate((minxy, maxxy), axis=1)
        return trans_boxes

    def apply_rotated_box(self, rotated_box: np.ndarray):
        """
        Apply the transform on rotated boxes.
        Args:
            rotated_box (ndarray): Nx5 floating point array of
                (x_center, y_center, width, height, angle_degrees) format
                in absolute coordinates.
        Returns:
            ndarray: rotated box after apply the transformation.
        Note:
            The coordinates are not pixel indices. Coordinates inside an image of
            shape (H, W) are in range [0, W] or [0, H].
            This function does not clip boxes to force them inside the image.
            It is up to the application that uses the boxes to decide.
        """
        raise NotImplementedError()

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply the transform on a list of polygons, each represented by a Nx2
        array. By default will just transform all the points.
        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            list[ndarray]: polygon after apply the transformation.
        Note:
            The coordinates are not pixel indices. Coordinates on an image of
            shape (H, W) are in range [0, W] or [0, H].
        """
        return [self.apply_coords(p) for p in polygons]

    @classmethod
    def register_type(cls, data_type: str, func: Optional[Callable] = None):
        """
        Register the given function as a handler that this transform will use
        for a specific data type.
        Args:
            data_type (str): the name of the data type (e.g., box)
            func (callable): takes a transform and a data, returns the
                transformed data.
        Examples:
        .. code-block:: python
            # call it directly
            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            HFlipTransform.register_type("voxel", func)
            # or, use it as a decorator
            @HFlipTransform.register_type("voxel")
            def func(flip_transform, voxel_data):
                return transformed_voxel_data
            # ...
            transform = HFlipTransform(...)
            transform.apply_voxel(voxel_data)  # func will be called
        """
        if func is None:  # the decorator style

            def wrapper(decorated_func):
                assert decorated_func is not None
                cls.register_type(data_type, decorated_func)
                return decorated_func

            return wrapper

        assert callable(
            func
        ), "You can only register a callable to a Transform. Got {} instead.".format(
            func
        )
        argspec = inspect.getfullargspec(func)
        assert len(argspec.args) == 2, (
            "You can only register a function that takes two positional "
            "arguments to a Transform! Got a function with spec {}".format(str(argspec))
        )
        setattr(cls, "apply_" + data_type, func)

    def inverse(self) -> "Transform":
        """
        Create a transform that inverts the geometric changes (i.e. change of
        coordinates) of this transform.
        Note that the inverse is meant for geometric changes only.
        The inverse of photometric transforms that do not change coordinates
        is defined to be a no-op, even if they may be invertible.
        Returns:
            Transform:
        """
        raise NotImplementedError

    def __repr__(self):
        """
        Produce something like:
        "MyTransform(field1={self.field1}, field2={self.field2})"
        """
        try:
            sig = inspect.signature(self.__init__)
            classname = type(self).__name__
            argstr = []
            for name, param in sig.parameters.items():
                assert (
                        param.kind != param.VAR_POSITIONAL
                        and param.kind != param.VAR_KEYWORD
                ), "The default __repr__ doesn't support *args or **kwargs"
                assert hasattr(self, name), (
                    "Attribute {} not found! "
                    "Default __repr__ only works if attributes match the constructor.".format(
                        name
                    )
                )
                attr = getattr(self, name)
                default = param.default
                if default is attr:
                    continue
                attr_str = pprint.pformat(attr)
                if "\n" in attr_str:
                    # don't show it if pformat decides to use >1 lines
                    attr_str = "..."
                argstr.append("{}={}".format(name, attr_str))
            return "{}({})".format(classname, ", ".join(argstr))
        except AssertionError:
            return super().__repr__()


_T = TypeVar("_T")


class IAATransform(Transform):
    """
    TODO: Add wrapper for imgaug Augmentations.
    """
    pass


# pyre-ignore-all-errors
class TransformList(Transform):
    """
    Maintain a list of transform operations which will be applied in sequence.
    Attributes:
        transforms (list[Transform])
    """

    def __init__(self, transforms: List[Transform]):
        """
        Args:
            transforms (list[Transform]): list of transforms to perform.
        """
        super().__init__()
        # "Flatten" the list so that TransformList do not recursively contain TransfomList.
        # The additional hierarchy does not change semantic of the class, but cause extra
        # complexities in e.g, telling whether a TransformList contains certain Transform
        tfms_flatten = []
        for t in transforms:
            assert isinstance(
                t, Transform
            ), f"TransformList requires a list of Transform. Got type {type(t)}!"
            if isinstance(t, TransformList):
                tfms_flatten.extend(t.transforms)
            else:
                tfms_flatten.append(t)
        self.transforms = tfms_flatten

    def _apply(self, x: _T, meth: str) -> _T:
        """
        Apply the transforms on the input.
        Args:
            x: input to apply the transform operations.
            meth (str): meth.
        Returns:
            x: after apply the transformation.
        """
        for t in self.transforms:
            x = getattr(t, meth)(x)
        return x

    def __getattribute__(self, name: str):
        # use __getattribute__ to win priority over any registered dtypes
        if name.startswith("apply_"):
            return lambda x: self._apply(x, name)
        return super().__getattribute__(name)

    def __add__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(self.transforms + others)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        self.transforms.extend(others)
        return self

    def __radd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        others = other.transforms if isinstance(other, TransformList) else [other]
        return TransformList(others + self.transforms)

    def __len__(self) -> int:
        """
        Returns:
            Number of transforms contained in the TransformList.
        """
        return len(self.transforms)

    def __getitem__(self, idx) -> Transform:
        return self.transforms[idx]

    def inverse(self) -> "TransformList":
        """
        Invert each transform in reversed order.
        """
        return TransformList([x.inverse() for x in self.transforms[::-1]])

    def __repr__(self) -> str:
        msgs = [str(t) for t in self.transforms]
        return "TransformList[{}]".format(", ".join(msgs))

    __str__ = __repr__

    # The actual implementations are provided in __getattribute__.
    # But abstract methods need to be declared here.
    def apply_image(self, x):
        raise NotImplementedError


class HFlipTransform(Transform):
    """
    Perform horizontal flip.
    """

    def __init__(self, width: Optional[int] = None):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        # NOTE: opencv would be faster:
        # https://github.com/pytorch/pytorch/issues/16424#issuecomment-580695672
        if img.ndim <= 3:  # HxW, HxWxC
            self._width = img.shape[1] if self.width is None else None
            return np.flip(img, axis=1)
        else:  # NxHxWxC
            self._width = img.shape[-2] if self.width is None else None
            return np.flip(img, axis=-2)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        assert self.width is not None or self._width is not None, \
            "image transform must be applied before coordinate transforms if width is not provided"
        coords[:, 0] = (self.width if self.width else self._width) - coords[:, 0]
        return coords

    def apply_rotated_box(self, rotated_box: np.ndarray) -> np.ndarray:
        """
        Apply the horizontal flip transform on rotated boxes.
        Args:
            rotated_box (ndarray): Nx5 floating point array of
                (x_center, y_center, width, height, angle_degrees) format
                in absolute coordinates.
        """
        assert self.width is not None or self._width is not None, \
            "image transform must be applied before rotated_box transforms if width is not provided"
        # Transform x_center
        rotated_box[:, 0] = (self.width if self.width else self._width) - rotated_box[:, 0]
        # Transform angle
        rotated_box[:, 4] = -rotated_box[:, 4]
        return rotated_box

    def inverse(self) -> Transform:
        """
        The inverse is to flip again
        """
        return self


class VFlipTransform(Transform):
    """
    Perform vertical flip.
    """

    def __init__(self, height: Optional[int] = None):
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Flip the image(s).
        Args:
            img (ndarray): of shape HxW, HxWxC, or NxHxWxC. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: the flipped image(s).
        """
        tensor = torch.from_numpy(np.ascontiguousarray(img))
        if len(tensor.shape) == 2:
            # For dimension of HxW.
            self._height = tensor.size(0) if self.height is None else None
            tensor = tensor.flip((-2))
        elif len(tensor.shape) > 2:
            # For dimension of HxWxC, NxHxWxC.
            self._height = tensor.size(-3) if self.height is None else None
            tensor = tensor.flip((-3))
        return tensor.numpy()

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Flip the coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: the flipped coordinates.
        Note:
            The inputs are floating point coordinates, not pixel indices.
            Therefore they are flipped by `(W - x, H - y)`, not
            `(W - 1 - x, H - 1 - y)`.
        """
        assert self.height is not None or self._height is not None, \
            "image transform must be applied before rotated_box transforms if height is not provided"
        coords[:, 1] = (self.height if self.height else self._height) - coords[:, 1]
        return coords

    def apply_rotated_box(self, rotated_box: np.ndarray) -> np.ndarray:
        """
        Apply the horizontal flip transform on rotated boxes.
        Args:
            rotated_box (ndarray): Nx5 floating point array of
                (x_center, y_center, width, height, angle_degrees) format
                in absolute coordinates.
        """
        assert self.height is not None or self._height is not None, \
            "image transform must be applied before rotated_box transforms if height is not provided"
        # Transform x_center
        rotated_box[:, 1] = (self.height if self.height else self._height) - rotated_box[:, 1]
        # Transform angle
        rotated_box[:, 4] = -rotated_box[:, 4]
        return rotated_box

    def inverse(self) -> Transform:
        """
        The inverse is to flip again
        """
        return self


class NoOpTransform(Transform):
    """
    A transform that does nothing.
    """

    def __init__(self):
        super().__init__()

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def inverse(self) -> Transform:
        return self

    def __getattr__(self, name: str):
        if name.startswith("apply_"):
            return lambda x: x
        raise AttributeError("NoOpTransform object has no attribute {}".format(name))


class ScaleTransform(Transform):
    """
    Resize the image to a target size.
    """

    def __init__(self, new_h: int, new_w: int, h: Optional[int] = None, w: Optional[int] = None,
                 interp: str = 'bilinear'):
        """
        Args:
            new_h, new_w (int): new image size.
            h, w (Optional[int]): original image size, if not provided, will be inferred from apply_image
            interp (str): interpolation methods. Options includes `nearest`, `linear`
                (3D-only), `bilinear`, `bicubic` (4D-only), and `area`. Default value is `bilinear`.
                Details can be found in:
                https://pytorch.org/docs/stable/nn.functional.html
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Resize the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options includes `nearest`, `linear`
                (3D-only), `bilinear`, `bicubic` (4D-only), and `area`.
                Details can be found in:
                https://pytorch.org/docs/stable/nn.functional.html
        Returns:
            ndarray: resized image(s).
        """
        if len(img.shape) == 4:
            h, w = img.shape[1:3]
        elif len(img.shape) in (2, 3):
            h, w = img.shape[:2]
        else:
            raise ("Unsupported input with shape of {}".format(img.shape))
        self._h = h if self.h is None else None
        self._w = w if self.w is None else None
        assert (
                (not self.h and not self.w) or
                self.h == h and self.w == w
        ), "Input size mismatch h w {}:{} -> {}:{}".format(self.h, self.w, h, w)
        interp_method = interp if interp is not None else self.interp
        # Option of align_corners is only supported for linear, bilinear,
        # and bicubic.
        if interp_method in ["linear", "bilinear", "bicubic"]:
            align_corners = False
        else:
            align_corners = None

        # note: this is quite slow for int8 images because torch does not
        # support it https://github.com/pytorch/pytorch/issues/5580
        float_tensor = torch.nn.functional.interpolate(
            to_float_tensor(img),
            size=(self.new_h, self.new_w),
            mode=interp_method,
            align_corners=align_corners,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Compute the coordinates after resize.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: resized coordinates.
        """
        coords[:, 0] = coords[:, 0] * (self.new_w * 1.0 / (self.w if self.w else self._w))
        coords[:, 1] = coords[:, 1] * (self.new_h * 1.0 / (self.h if self.h else self._h))
        return coords

    def apply_rotated_box(self, rotated_boxes):
        """
        Apply the resizing transform on rotated boxes. For details of how these (approximation)
        formulas are derived, please refer to :meth:`RotatedBoxes.scale`.
        Args:
            rotated_boxes (ndarray): Nx5 floating point array of
                (x_center, y_center, width, height, angle_degrees) format
                in absolute coordinates.
        """
        scale_factor_x = self.new_w * 1.0 / (self.w if self.w else self._w)
        scale_factor_y = self.new_h * 1.0 / (self.h if self.h else self._h)
        rotated_boxes[:, 0] *= scale_factor_x
        rotated_boxes[:, 1] *= scale_factor_y
        theta = rotated_boxes[:, 4] * np.pi / 180.0
        c = np.cos(theta)
        s = np.sin(theta)
        rotated_boxes[:, 2] *= np.sqrt(np.square(scale_factor_x * c) + np.square(scale_factor_y * s))
        rotated_boxes[:, 3] *= np.sqrt(np.square(scale_factor_x * s) + np.square(scale_factor_y * c))
        rotated_boxes[:, 4] = np.arctan2(scale_factor_x * s, scale_factor_y * c) * 180 / np.pi

        return rotated_boxes

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply resize on the full-image segmentation.
        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: resized segmentation.
        """
        segmentation = self.apply_image(segmentation, interp="nearest")
        return segmentation

    def inverse(self) -> Transform:
        """
        The inverse is to resize it back.
        """
        return ScaleTransform((self.h if self.h else self._h), (self.w if self.w else self._w), self.new_h, self.new_w,
                              self.interp)


class GridSampleTransform(Transform):
    def __init__(self, grid: np.ndarray, interp: str):
        """
        Args:
            grid (ndarray): grid has x and y input pixel locations which are
                used to compute output. Grid has values in the range of [-1, 1],
                which is normalized by the input height and width. The dimension
                is `N x H x W x 2`.
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply grid sampling on the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): interpolation methods. Options include `nearest` and
                `bilinear`.
        Returns:
            ndarray: grid sampled image(s).
        """
        interp_method = interp if interp is not None else self.interp
        float_tensor = torch.nn.functional.grid_sample(
            to_float_tensor(img),  # NxHxWxC -> NxCxHxW.
            torch.from_numpy(self.grid),
            mode=interp_method,
            padding_mode="border",
            align_corners=False,
        )
        return to_numpy(float_tensor, img.shape, img.dtype)

    def apply_coords(self, coords: np.ndarray):
        """
        Not supported.
        """
        raise NotImplementedError()

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply grid sampling on the full-image segmentation.
        Args:
            segmentation (ndarray): of shape HxW. The array should have integer
                or bool dtype.
        Returns:
            ndarray: grid sampled segmentation.
        """
        segmentation = self.apply_image(segmentation, interp="nearest")
        return segmentation


class CropTransform(Transform):
    def __init__(self, x0: int, y0: int, x1: int, y1: int):
        """
        Args:
            x0, y0, x1, y1 (int): crop the image(s) by img[y0:y1+1, x0:y1+1]. Note that x0, y0, x1, y1 are image indices,
            i.e., x0, x1 is in [0, W-1] and y0, y1 in [0, H-1].
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Crop the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
        Returns:
            ndarray: cropped image(s).
        """
        if len(img.shape) <= 3:
            return img[self.y0: self.y1 + 1, self.x0: self.x1 + 1]
        else:
            return img[..., self.y0: self.y1, self.x0: self.x1, :]

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply crop transform on coordinates.
        Args:
            coords (ndarray): floating point array of shape Nx2. Each row is
                (x, y).
        Returns:
            ndarray: cropped coordinates.
        """
        coords[:, 0] -= self.x0
        coords[:, 1] -= self.y0
        return coords

    def apply_polygons(self, polygons: list) -> list:
        """
        Apply crop transform on a list of polygons, each represented by a Nx2 array.
        It will crop the polygon with the box, therefore the number of points in the
        polygon might change.
        Args:
            polygon (list[ndarray]): each is a Nx2 floating point array of
                (x, y) format in absolute coordinates.
        Returns:
            ndarray: cropped polygons.
        """
        import shapely.geometry as geometry

        # Create a window that will be used to crop
        crop_box = geometry.box(
            self.x0, self.y0, self.x1 + 1, self.y1 + 1
        ).buffer(0.0)

        cropped_polygons = []

        for polygon in polygons:
            polygon = geometry.Polygon(polygon).buffer(0.0)
            # polygon must be valid to perform intersection.
            assert polygon.is_valid, polygon
            cropped = polygon.intersection(crop_box)
            if cropped.is_empty:
                continue
            if not isinstance(cropped, geometry.collection.BaseMultipartGeometry):
                cropped = [cropped]
            # one polygon may be cropped to multiple ones
            for poly in cropped:
                # It could produce lower dimensional objects like lines or
                # points, which we want to ignore
                if not isinstance(poly, geometry.Polygon) or not poly.is_valid:
                    continue
                coords = np.asarray(poly.exterior.coords)
                # NOTE This process will produce an extra identical vertex at
                # the end. So we remove it. This is tested by
                # `tests/test_data_transform.py`
                cropped_polygons.append(coords[:-1])
        return [self.apply_coords(p) for p in cropped_polygons]


class ExtentTransform(Transform):
    """
    Extracts a subregion from the source image and scales it to the output size.
    The fill color is used to map pixels from the source rect that fall outside
    the source image.
    See: https://pillow.readthedocs.io/en/latest/PIL.html#PIL.ImageTransform.ExtentTransform
    """

    def __init__(self, src_rect, output_size, interp=Image.LINEAR, fill=0):
        """
        Args:
            src_rect (x0, y0, x1, y1): src coordinates
            output_size (h, w): dst image size
            interp: PIL interpolation methods
            fill: Fill color used when src_rect extends outside image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        h, w = self.output_size
        if len(img.shape) > 2 and img.shape[2] == 1:
            pil_image = Image.fromarray(img[:, :, 0], mode="L")
        else:
            pil_image = Image.fromarray(img)
        pil_image = pil_image.transform(
            size=(w, h),
            method=Image.EXTENT,
            data=self.src_rect,
            resample=interp if interp else self.interp,
            fill=self.fill,
        )
        ret = np.asarray(pil_image)
        if len(img.shape) > 2 and img.shape[2] == 1:
            ret = np.expand_dims(ret, -1)
        return ret

    def apply_coords(self, coords):
        # Transform image center from source coordinates into output coordinates
        # and then map the new origin to the corner of the output image.
        h, w = self.output_size
        x0, y0, x1, y1 = self.src_rect
        new_coords = coords.astype(np.float32)
        new_coords[:, 0] -= 0.5 * (x0 + x1)
        new_coords[:, 1] -= 0.5 * (y0 + y1)
        new_coords[:, 0] *= w / (x1 - x0)
        new_coords[:, 1] *= h / (y1 - y0)
        new_coords[:, 0] += 0.5 * w
        new_coords[:, 1] += 0.5 * h
        return new_coords

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=Image.NEAREST)
        return segmentation


class RotationTransform(Transform):
    """
    This method returns a copy of this image, rotated the given
    number of degrees counter clockwise around its center.
    """

    def __init__(self, h, w, angle, expand=True, center=None, interp=None):
        """
        Args:
            h, w (int): original image size
            angle (float): degrees for rotation
            expand (bool): choose if the image should be resized to fit the whole
                rotated image (default), or simply cropped
            center (tuple (width, height)): coordinates of the rotation center
                if left to None, the center will be fit to the center of each image
                center has no effect if expand=True because it only affects shifting
            interp: cv2 interpolation method, default cv2.INTER_LINEAR
        """
        super().__init__()
        image_center = np.array((w / 2, h / 2))
        if center is None:
            center = image_center
        if interp is None:
            interp = cv2.INTER_LINEAR
        abs_cos, abs_sin = (abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle))))
        if expand:
            # find the new width and height bounds
            bound_w, bound_h = np.rint(
                [h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]
            ).astype(int)
        else:
            bound_w, bound_h = w, h

        self._set_attributes(locals())
        self.rm_coords = self.create_rotation_matrix()
        # Needed because of this problem https://github.com/opencv/opencv/issues/11784
        self.rm_image = self.create_rotation_matrix(offset=-0.5)

    def apply_image(self, img, interp=None):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0 or self.angle % 360 == 0:
            return img
        assert img.shape[:2] == (self.h, self.w)
        interp = interp if interp is not None else self.interp
        return cv2.warpAffine(img, self.rm_image, (self.bound_w, self.bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y) points
        """
        coords = np.asarray(coords, dtype=float)
        if len(coords) == 0 or self.angle % 360 == 0:
            return coords
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def apply_segmentation(self, segmentation):
        segmentation = self.apply_image(segmentation, interp=cv2.INTER_NEAREST)
        return segmentation

    def create_rotation_matrix(self, offset=0):
        center = (self.center[0] + offset, self.center[1] + offset)
        rm = cv2.getRotationMatrix2D(tuple(center), self.angle, 1)
        if self.expand:
            # Find the coordinates of the center of rotation in the new image
            # The only point for which we know the future coordinates is the center of the image
            rot_im_center = cv2.transform(self.image_center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([self.bound_w / 2, self.bound_h / 2]) + offset - rot_im_center
            # shift the rotation center to the new coordinates
            rm[:, 2] += new_center
        return rm

    def inverse(self):
        """
        The inverse is to rotate it back with expand, and crop to get the original shape.
        """
        if not self.expand:  # Not possible to inverse if a part of the image is lost
            raise NotImplementedError()
        rotation = RotationTransform(
            self.bound_h, self.bound_w, -self.angle, True, None, self.interp
        )
        crop = CropTransform(
            (rotation.bound_w - self.w) // 2, (rotation.bound_h - self.h) // 2, self.w, self.h
        )
        return TransformList([rotation, crop])


class PhotometricTransform(Transform):
    """
    Generic wrapper for any photometric transforms.
    These transformations should only affect the color space and
    not the coordinate space of the image (e.g. annotation
    coordinates such as bounding boxes should not be changed)
    The input image should assumed to be of type uint8 in range [0, 255] and RGB order.
    The shape can be NxHxWxC, or HxWxC or HxW.
    """

    def __init__(self, op: Optional[Callable] = None):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in an ndarray and returns an ndarray.
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img):
        if self.op is None:
            raise NotImplementedError()
        return self.op(img)

    def apply_coords(self, coords):
        return coords

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        return box

    def apply_rotated_box(self, rotated_box: np.ndarray):
        return rotated_box

    def apply_polygons(self, polygons: list) -> list:
        return polygons

    def inverse(self):
        return NoOpTransform()

    def apply_segmentation(self, segmentation):
        return segmentation


class PILPhotometricTransform(PhotometricTransform):
    """
    Generic wrapper for PIL Photometric image transforms,
        which affect the color space and not the coordinate
        space of the image
    """

    def __init__(self, op: Callable):
        """
        Args:
            op (Callable): operation to be applied to the image,
                which takes in a PIL Image and returns a transformed
                PIL Image.
                For reference on possible operations see:
                - https://pillow.readthedocs.io/en/stable/
        """
        if not callable(op):
            raise ValueError("op parameter should be callable")
        super().__init__(op)

    def apply_image(self, img):
        img = Image.fromarray(img)
        return np.asarray(super().apply_image(img))


class BrightnessTransform(PhotometricTransform):
    """
    Adjust image brightness.
    This transform controls the brightness of an image. An
    enhancement factor of 0.0 gives a black image.
    A factor of 1.0 gives the original image. This function
    blends the source image and the degenerated black image:
    ``output = img * factor + degenerated * (1 - factor)``
    """

    def __init__(self, factor: float = 1.):
        """
        Args:
            factor (float): A value controls the enhancement.
                Factor 1.0 returns the original image, lower
                factors mean less color (brightness, contrast,
                etc), and higher values more. Default 1.
        """
        super().__init__(partial(adjust_brightness, factor=factor))


class ContrastTransform(PhotometricTransform):
    """
    Adjust image contrast.
    This transform controls the contrast of an image. An
    enhancement factor of 0.0 gives a solid grey
    image. A factor of 1.0 gives the original image. It
    blends the source image and the degenerated mean image:
    ``output = img * factor + degenerated * (1 - factor)``
    """

    def __init__(self, factor: float = 1.):
        """
        Args:
            factor (float): A value controls the enhancement.
                Factor 1.0 returns the original image, lower
                factors mean less color (brightness, contrast,
                etc), and higher values more. Default 1.
        """
        super().__init__(partial(adjust_contrast, factor=factor))


class ColorTransform(PhotometricTransform):
    """
    This transform blends the source image and its gray image:
    ``output = img * alpha + gray_img * beta + gamma``
        """

    def __init__(
            self,
            alpha: Optional[Union[int, float]] = 1.,
            beta: Optional[Union[int, float]] = None,
            gamma: Optional[Union[int, float]] = 0
    ):
        """
        Args:
            alpha (int | float): Weight for the source image. Default 1.
            beta (int | float): Weight for the converted gray image.
                If None, it's assigned the value (1 - `alpha`).
            gamma (int | float): Scalar added to each sum.
                Same as :func:`cv2.addWeighted`. Default 0.
        """
        super().__init__(partial(adjust_color, alpha=alpha, beta=beta, gamma=gamma))


class BlendTransform(PhotometricTransform):
    """
    Blend target images with the given source image.
    """

    def __init__(self, src_image: np.ndarray, src_weight: float, dst_weight: float, bias: float = 0.):
        """
        Blends the input image (dst_image) with the src_image using formula:
        ``src_weight * src_image + dst_weight * dst_image``
        Args:
            src_image (ndarray): Input image is blended with this image
            src_weight (float): Blend weighting of src_image
            dst_weight (float): Blend weighting of dst_image
            bias (float): Blend bias to be added on the output image
        """
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray, interp: str = None) -> np.ndarray:
        """
        Apply blend transform on the image(s).
        Args:
            img (ndarray): of shape NxHxWxC, or HxWxC or HxW. The array can be
                of type uint8 in range [0, 255], or floating point in range
                [0, 1] or [0, 255].
            interp (str): keep this option for consistency, perform blend would not
                require interpolation.
        Returns:
            ndarray: blended image(s).
        """
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = self.src_weight * self.src_image + self.dst_weight * img + self.bias
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return self.src_weight * self.src_image + self.dst_weight * img + self.bias

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the coordinates.
        """
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply no transform on the full-image segmentation.
        """
        return segmentation

    def inverse(self) -> Transform:
        """
        The inverse is a no-op.
        """
        return NoOpTransform()
