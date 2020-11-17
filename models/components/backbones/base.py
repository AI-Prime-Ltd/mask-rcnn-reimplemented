import inspect
import logging
import math
from copy import deepcopy
from functools import partial
from typing import Dict, Iterable, Optional, Callable

import torch.nn as nn
import torch.utils.hooks as hooks
import torch.utils.model_zoo as model_zoo

from .helpers import extract_layer

_logger = logging.getLogger(__name__)


def _variant(config: dict, meta: dict):
    return dict(config=config, meta=meta)


class MetaBackboneMeta(type):
    def build_model(
            cls: "MetaBackboneBase",
            variant: str,
            pretrained: bool = False,
            *,
            pretrained_filter_fn: Optional[Callable] = None,
            pretrained_map_location: str = "cpu",
            pretrained_strict: bool = True,
            pretrained_auto_channel_cvt: bool = True,
            pretrained_auto_classifier_cvt: bool = True,
            **kwargs
    ):
        # load default config for the required variant
        config = deepcopy(cls.variants[variant]["config"])
        meta = cls.variants[variant]["meta"]
        # update default config with custom config
        config.update(kwargs)
        # init model with config
        model = cls(variant, **config)

        # for classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
        if pretrained:
            if "url" not in meta.keys() or not meta["url"]:
                _logger.warning(
                    f"{variant} has no pretrained model specified in meta, skipped loading."
                )
                return model
            model.load_pretrained(
                pretrained_filter_fn,
                pretrained_map_location,
                pretrained_strict,
                auto_channel_cvt=pretrained_auto_channel_cvt,
                auto_classifier_cvt=pretrained_auto_classifier_cvt
            )

        return model

    def _create_variant(cls, variant, pretrained=False, **kwargs):
        return cls.build_model(
            variant, pretrained,
            **kwargs
        )

    def __getattr__(cls, item):
        if item.startswith("create_"):
            variant = item[7:]
            if variant in cls.variants.keys():
                return partial(cls._create_variant, variant)
        raise AttributeError()


class MetaBackboneBase(nn.Module, metaclass=MetaBackboneMeta):
    """
    Interface for all backbone models.
    """
    """
    Different variants and the corresponding configs to initialize them should be specified here.
    """
    variants: Dict[str, Dict]

    def __new__(cls, *args, **kwargs):
        inst = super().__new__(cls)
        # replace the original __init__ method with no-op
        if not hasattr(cls, "__ori_init__"):
            cls.__ori_init__ = cls.__init__
            cls.__init__ = lambda *args, **kwargs: None

        # call the original __init__ method before registering hook
        inst.__ori_init__(*args, **kwargs)
        for f in set(inst._out_features or tuple()):
            try:
                m = extract_layer(inst, f)
            except (AttributeError, KeyError) as e:
                _logger.exception(
                    f"cannot find submodule: {f}"
                )
                raise e
            else:
                m.register_forward_hook(partial(inst._collect_feature_hook, f))
        return inst

    def __init__(
            self,
            variant: str, *,
            in_channels: int = 3, out_features: Optional[Iterable[str]] = None,
            num_classes: Optional[int] = 1000
    ):
        super(MetaBackboneBase, self).__init__()
        self._config = {}
        self._meta = deepcopy(self.variants[variant]["meta"])
        self.variant = variant
        self.out_features = {}
        self._out_features = out_features
        self._load_config()

    def _load_config(self, ignored_locals=("self", "variant", "out_features"), expose_in_self=True, use_deepcopy=False):
        """
        This function is expected to be directly called in __init__ as a helper function to easily port all
        arguments into current instance
        """
        last_locals = inspect.currentframe().f_back.f_locals
        for k, v in last_locals.items():
            if k.startswith("_"):
                continue
            if k in ignored_locals:
                continue
            if use_deepcopy:
                v = deepcopy(v)
            self._config[k] = v
            if expose_in_self:
                setattr(type(self), k, property(lambda self, k=k: self._config[k]))
        return self._config

    @property
    def config(self) -> Dict:
        return self._config

    @property
    def meta(self) -> Dict:
        return self._meta

    def _collect_feature_hook(self, name, _, __, output):
        self.out_features[name] = output
        return None

    def register_forward_hook(self, hook: Callable[..., None]) -> hooks.RemovableHandle:
        _logger.warning(
            "tring to register a hook to an initialized model, note that the output_features may affected by the"
            "hook if the hook doesn't apply in-place ops to the output_features."
        )
        return super(MetaBackboneBase, self).register_forward_hook(hook)

    def load_pretrained(
            self,
            filter_fn: Optional[Callable] = None,
            map_location: str = "cpu",
            strict: bool = True,
            *,
            auto_channel_cvt: bool = True,
            auto_classifier_cvt: bool = True
    ):
        default_cfg = self.variants[self.variant]["config"]
        meta = self.variants[self.variant]["meta"]
        if self.variant not in self.variants.keys() or not meta["url"]:
            _logger.warning(f"{self.variant} does not have pretrained model, using random initialization.")
            return self

        state_dict = model_zoo.load_url(meta["url"], progress=True, map_location=map_location)

        # filter state dict if necessary
        if filter_fn is not None:
            state_dict = filter_fn(state_dict)

        in_channels = self.config["in_channels"]
        default_in_channels = default_cfg["in_channels"]
        # perform automatically weight converting when in_channels != default_in_channels
        if in_channels != default_in_channels:
            assert "first_conv" in meta, (
                "'first_conv' must be specified in variant meta when in_channels is not consistent between current"
                " configured model and the pretrained model."
            )
            conv1_name = meta['first_conv']
            conv1_weight = state_dict[conv1_name + '.weight']
            conv1_type = conv1_weight.dtype
            O, I, J, K = conv1_weight.shape
            conv1_weight = conv1_weight.float()

            if auto_channel_cvt:
                _logger.warning(
                    f"in_channels inconsistency detected, "
                    f"converting first_conv ({conv1_name}) pretrained weights "
                    f"from {default_in_channels} to {in_channels} channel."
                )
                if in_channels == 1:
                    # Some weights are in torch.half, ensure it's float for sum on CPU
                    if I > 3 and conv1_weight.shape[1] % 3 == 0:
                        # For models with space2depth stems
                        conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
                        conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
                    else:
                        conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
                    conv1_weight = conv1_weight.to(conv1_type)
                    state_dict[conv1_name + '.weight'] = conv1_weight
                elif in_channels != 3 and I == 3:
                    # NOTE this strategy should be better than random init, but there could be other combinations of
                    # the original RGB input layer weights that'd work better for specific cases.
                    _logger.info('Repeating first conv (%s) weights in channel dim.' % conv1_name)
                    repeat = int(math.ceil(in_channels / 3))
                    conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[:, :in_channels, :, :]
                    conv1_weight *= (3 / float(in_channels))
                    conv1_weight = conv1_weight.to(conv1_type)
                    state_dict[conv1_name + '.weight'] = conv1_weight
                else:
                    _logger.warning(
                        'auto converting failed, deleting first conv (%s) from pretrained weights.' % conv1_name
                    )
                    del state_dict[conv1_name + '.weight']
                    strict = False
            else:
                _logger.warning('deleting first conv (%s) from pretrained weights.' % conv1_name)
                del state_dict[conv1_name + '.weight']
                strict = False

        # if this backbone is pretrained as a classifier, check num_classes
        num_classes = self.config.get("num_classes", None)
        default_num_classes = default_cfg.get("num_classes", None)
        if not default_num_classes:
            if "classifier" in meta.keys():
                _logger.info(
                    f"deleting classifier ({meta['classifier']}) weights in pretrained model."
                )
                del state_dict[meta['classifier'] + '.weight']
                del state_dict[meta['classifier'] + '.bias']
                strict = False
        elif default_num_classes != num_classes and auto_classifier_cvt and default_num_classes - num_classes == 1:
            # special case for imagenet trained models with extra background class in pretrained weights
            classifier_name = meta['classifier']
            classifier_weight = state_dict[classifier_name + '.weight']
            state_dict[classifier_name + '.weight'] = classifier_weight[1:]
            classifier_bias = state_dict[classifier_name + '.bias']
            state_dict[classifier_name + '.bias'] = classifier_bias[1:]
        elif default_num_classes != num_classes:
            # completely discard fully connected for all other differences between pretrained and created model
            _logger.warning(
                f"deleting classifier ({meta['classifier']}) weights in pretrained model."
            )
            classifier_name = meta['classifier']
            del state_dict[classifier_name + '.weight']
            del state_dict[classifier_name + '.bias']
            strict = False

        self.load_state_dict(state_dict, strict=strict)


class MetaClassifierBase(MetaBackboneBase):
    def forward_classifier(self, x):
        if self.num_classes:
            raise NotImplementedError()

    def forward_features(self, x):
        raise NotImplementedError()

    def forward(self, x):
        out = {}
        if self.num_classes:
            out["classifier"] = self.forward_classifier(x)
        else:
            out["features"] = self.forward_features(x)
        out.update(self.out_features)

        return out


class BackboneBase(MetaBackboneBase):
    def __init__(
            self,
            in_channels: int = 3, out_features: Iterable = ("classifier",),
            num_classes: Optional[int] = 1000
    ):
        super().__init__(
            self.__class__.__name__.lower(),
            in_channels=in_channels,
            out_features=out_features,
            num_classes=num_classes
        )

    @classmethod
    def build_model(
            cls: "BackboneBase",
            pretrained: bool = False,
            *,
            pretrained_filter_fn: Optional[Callable] = None,
            pretrained_map_location: str = "str",
            pretrained_strict: bool = True,
            pretrained_auto_channel_cvt: bool = True,
            pretrained_auto_classifier_cvt: bool = True,
            **kwargs
    ):
        return super().build_model(
            variant=cls.__name__.lower(),
            pretrained=pretrained,
            pretrained_filter_fn=pretrained_filter_fn,
            pretrained_map_location=pretrained_map_location,
            pretrained_strict=pretrained_strict,
            pretrained_auto_channel_cvt=pretrained_auto_channel_cvt,
            pretrained_auto_classifier_cvt=pretrained_auto_classifier_cvt
        )
