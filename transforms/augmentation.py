"""
Perform random transform
"""
from random import Random
from typing import Optional, Dict

from .transform import NoOpTransform, Transform

_rnd = Random()


def seed(seed=1024):
    global _rnd
    _rnd = Random(seed)


class Augmentation(object):
    def __init__(
            self,
            transform: Optional[Transform] = None,
            transform_kwargs: Optional[Dict] = None,
            prob: Optional[float] = None
    ):
        self.transform = transform or NoOpTransform
        self.tkwarg = transform_kwargs or dict()
        self.prob = prob or 1.

    def apply(self, *imgs, **kwargs):
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
            >>> img1, img2 = augmentation.apply(img1, img2)
            >>>
            >>> # apply transform to two images (image batches) and coordinates
            >>> transformed = augmentation.apply(image=(img1, img2), coords=(kpts, ))
            >>> img1 = transformed["image"][0]
            >>>
            >>> # transform images named 'img1', 'img2' and coordinates named 'kpts'.
            >>> transformed = augmentation.apply(
            >>>     image={"img1": img1, "img2": img2},
            >>>     coords={"kpts": kpts}
            >>> )
            >>> img1 = transformed["img1"]
            ```
        """
        if self.prob > _rnd.random():
            self.transform(**{k: v() for k, v in self.tkwarg}).apply(*imgs, **kwargs)
        else:
            return NoOpTransform().apply(*imgs, **kwargs)

    def __repr__(self):
        """
        Produce something like:
        "Augmentation(field1={self.field1}, field2={self.field2})"
        """
        fields = list("{}={}".format(k, v) for k, v in self.tkwarg.items())
        return f"""Augmentation({self.transform.__repr__()}, p={self.prob}, {", ".join(fields)})"""

    __str__ = __repr__


class Compose(Augmentation):
    def __init__(self, argument_list=[]):
        super(Compose, self).__init__()
        self.aug_list = []
        for aug in argument_list:
            self.aug_list.append(aug if isinstance(aug, Augmentation) else Augmentation(*aug))

    def apply(self, *imgs, **kwargs):
        for aug in self.aug_list:
            aug.apply(*imgs, **kwargs)

    def __add__(self, other: "Compose") -> "Compose":
        """
        Args:
            other (Compose): transformation to add.
        Returns:
            Compose: list of transforms.
        """
        return Compose(self.aug_list + other.aug_list)

    def __iadd__(self, other: "TransformList") -> "TransformList":
        """
        Args:
            other (TransformList): transformation to add.
        Returns:
            TransformList: list of transforms.
        """
        self.aug_list.extend(other.aug_list)
        return self

    def __repr__(self) -> str:
        msgs = [str(t) for t in self.aug_list]
        return "{}[{}]".format(self.__class__.__name__, ", ".join(msgs))


class RandomCompose(Compose):
    def apply(self, *imgs, **kwargs):
        for aug in _rnd.choices(self.aug_list, k=len(self.aug_list)):
            aug.apply(*imgs, **kwargs)


class Distribution(object):
    def _set_attributes(self, params=None) -> None:
        """
        Set attributes from the input list of parameters.
        Args:
            params (list): list of parameters.
        """

        if params:
            for k, v in params.items():
                if k != "self" and not k.startswith("_"):
                    setattr(self, k, v)

    def __repr__(self):
        return f"""{self.__class__.__name__}({", ".join(
            "{}={}".format(k, v) for k, v in self.__dict__.items() if (not k.startswith("_")) and (v is not None)
        )})"""

    __str__ = __repr__


class Constant(Distribution):
    def __init__(self, c=None):
        self._set_attributes(locals())

    def __call__(self):
        return self.c


class Uniform(Distribution):
    def __init__(self, a=0., b=1., cnt=None):
        self._set_attributes(locals())

    def __call__(self):
        if self.cnt is None:
            return _rnd.uniform(self.a, self.b)
        else:
            return tuple(_rnd.uniform(self.a, self.b) for _ in range(self.cnt))


class Gaussian(Distribution):
    def __init__(self, mu=0, sigma=1, cnt=None):
        self._set_attributes(locals())

    def __call__(self):
        if self.cnt is None:
            _rnd.gauss(self.mu, self.sigma)
        else:
            return tuple(_rnd.gauss(self.mu, self.sigma) for _ in range(self.cnt))
