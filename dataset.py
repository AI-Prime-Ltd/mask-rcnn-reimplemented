import contextlib
import hashlib
import io
import logging
import pickle
import time
from pathlib import Path
from typing import Optional, List, Union

import pycocotools.mask as mask_util
import pytorch_lightning as pl
import torch.utils.data as data
from tqdm import tqdm

from structures.boxes import BoxMode
from transforms.augmentation import Compose
from utils import Timer

logger = logging.getLogger(__name__)


def hash_file(p: Path):
    """
    Return md5 of a file.
    Args:
        p (Path): path to the file to be hashed
    Returns:
        str: md5 of the given file
    """
    BLOCK_SIZE = 65536  # The size of each read from the file
    file_hash = hashlib.md5()  # Create the hash object, can use something other than `.sha256()` if you wish
    with open(str(p), 'rb') as f:  # Open the file to read it's bytes
        fb = f.read(BLOCK_SIZE)  # Read from the file. Take in the amount declared above
        while len(fb) > 0:  # While there is still data being read from the file
            file_hash.update(fb)  # Update the hash
            fb = f.read(BLOCK_SIZE)  # Read the next block from the file
    return file_hash.hexdigest()


def load_coco_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """
    <ref: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py:72>
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.
    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., coco_2017_train).
            If provided, this function will also put "thing_classes" into
            the metadata associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.
    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO
    timer = Timer()
    json_file = Path(json_file)
    image_root = Path(image_root)
    assert json_file.is_file() and image_root.is_dir()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} with pycocotools takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # ----- Load COCO Categories ----- #
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)

    if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
        if "coco" not in dataset_name:
            logger.warning(
                """
                Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
                """
            )
    id_map = {v: i for i, v in enumerate(cat_ids)}

    # sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())

    # ---- Load COCO Images & Annotations ---- #
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    if "minival" not in json_file.name:
        # The popular valminusminival & minival annotations for COCO2014 contain this bug.
        # However the ratio of buggy annotations there is tiny and does not affect accuracy.
        # Therefore we explicitly white-list them.
        ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique!".format(
            json_file
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    # ---- Generate Dataset Dict ---- #
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0

    for (img_dict, anno_dict_list) in tqdm(imgs_anns, desc="parsing coco annotations"):
        record = {}
        record["file_name"] = str(image_root / img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        objs = []

        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.

            # The original COCO valminusminival2014 & minival2014 annotation files
            # actually contains bugs that, together with certain ways of using COCO API,
            # can trigger this assertion.
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'
            obj = {key: anno[key] for key in ann_keys if key in anno}
            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):  # RLE case
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:  # polygon case
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        logger.warning(
                            f"""
                            Invalid segmentation annotation found for image {anno['image_id']}.
                            """
                        )
                        continue  # ignore this instance

                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        # COCO's segmentation coordinates are floating points in [0, H or W],
                        # but keypoint coordinates are integers in [0, H-1 or W-1]
                        # Therefore we assume the coordinates are "pixel indices" and
                        # add 0.5 to convert to floating point coordinates.
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = id_map[obj["category_id"]]
            obj["category_info"] = cats[obj["category_id"]]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            "Filtered out {} instances without valid segmentation. ".format(
                num_instances_without_valid_segmentation
            )
            + "There might be issues in your dataset generation process. "
              "A valid polygon should be a list[float] with even length >= 6."
        )
    return dataset_dicts


class COCODataset(pl.LightningDataModule):
    def __init__(
            self, dataset_root, train_json, val_json=None, test_json=None,
            train_aug=None, val_aug=None, test_aug=None,
            used_cached_json=True, *coco_args, **coco_kwargs):
        super(COCODataset, self).__init__()
        self.dataset_root = Path(dataset_root)
        self.train_json = Path(train_json)
        self.train_json = self.train_json if self.train_json.is_absolute() else self.dataset_root / "annotations" / self.train_json
        self.val_json = None if val_json is None else Path(val_json)
        self.val_json = self.val_json if self.val_json and self.train_json.is_absolute() else self.dataset_root / "annotations" / self.val_json
        self.test_json = None if test_json is None else Path(test_json)
        self.test_json = self.test_json if self.test_json and self.test_json.is_absolute() else self.dataset_root / "annotations" / self.test_json
        self.train_aug = train_aug if train_aug else Compose()
        self.val_aug = val_aug if val_aug else Compose()
        self.test_aug = test_aug if test_aug else Compose()
        self.use_cached = used_cached_json
        self.coco_args = coco_args
        self.coco_kwargs = coco_kwargs

    def setup(self, stage: Optional[str] = None):
        if self.use_cached:
            for phase, p in {"train": self.train_json, "val": self.val_json, "test": self.test_json}.items():
                if p is None:
                    continue
                # TODO: come up with a better way to do this
                while True:
                    try:
                        Path("./cache/dataset.lock").touch(exist_ok=False)
                    except Exception as e:
                        time.sleep(0.1)
                    else:
                        break
                setattr(self, f"{phase}_dataset_md5", hash_file(p))
                src_md5 = getattr(self, f"{phase}_dataset_md5")
                for cached_file in Path("./cache/").glob(f"dataset-{src_md5}-*.pkl"):
                    dst_md5 = cached_file.name.split("-")[-1][:-5]
                    if dst_md5 == "temp":
                        Path("./cache/dataset.lock").unlink(missing_ok=True)
                        continue
                    elif hash_file(cached_file) == dst_md5:
                        logger.info(f"use cached file {cached_file.name} for json {p.name}")
                        with open(str(cached_file), "rb") as fp:
                            try:
                                dataset = pickle.load(fp)
                            except Exception as e:
                                logger.warning(f"failed to load {cached_file}: {e}. removing cached file.")
                                cached_file.unlink()
                                Path("./cache/dataset.lock").unlink(missing_ok=True)
                                break
                        setattr(self, f"{phase}_dataset", dataset)
                    else:
                        logger.warning(f"find broken cache {cached_file}, deleting...")
                        cached_file.unlink()
                Path("./cache/dataset.lock").unlink(missing_ok=True)

        # load dataset from original coco annotations and create caches
        for phase, p in {"train": self.train_json, "val": self.val_json, "test": self.test_json}.items():
            if p is None or hasattr(self, f"{phase}_dataset"):
                continue
            dataset = load_coco_json(str(getattr(self, f"{phase}_dataset")), self.dataset_root / phase, *self.coco_args,
                                     **self.coco_kwargs)
            setattr(self, f"{phase}_dataset", dataset)
            if self.use_cached:
                logger.info(f"creating cache for json {p.name}")
                if not hasattr(self, f"{phase}_dataset_md5"):
                    setattr(self, f"{phase}_dataset_md5", hash_file(p))
                src_md5 = getattr(self, f"{phase}_dataset_md5")
                # try to build new cache
                # TODO: come up with a better way to do this
                while True:
                    try:
                        Path("./cache/dataset.lock").touch(exist_ok=False)
                    except Exception as e:
                        time.sleep(0.1)
                    else:
                        break
                cached_file = Path("./cache/") / f"dataset-{src_md5}-temp.pkl"
                cached_file.unlink(missing_ok=True)
                cached_file.parent.mkdir(parents=True, exist_ok=True)
                with open(str(cached_file), "wb+") as fp:
                    # loadable since Python3.4
                    pickle.dump(dataset, fp, protocol=4)
                dst_md5 = hash_file(cached_file)
                cached_file.rename(cached_file.parent / f"dataset-{src_md5}-{dst_md5}.pkl")
                Path("./cache/dataset.lock").unlink(missing_ok=True)

    def train_dataloader(self, *args, **kwargs) -> data.DataLoader:
        pass

    def val_dataloader(self, *args, **kwargs) -> Union[data.DataLoader, List[data.DataLoader]]:
        pass

    def test_dataloader(self, *args, **kwargs) -> Union[data.DataLoader, List[data.DataLoader]]:
        pass


class COCOInstanceDataset(COCODataset):
    def __init__(self, dataset_root, train_aug=None, val_aug=None, test_aug=None, used_cached_json=True):
        super(COCOInstanceDataset, self).__init__(
            dataset_root,
            train_json="instances_train.json", val_json="instances_val.json", test_json=None,
            train_aug=train_aug, val_aug=val_aug, test_aug=test_aug,
            used_cached_json=used_cached_json
        )
        self.instances = []
        for img_info in self.train_dataset:
            image_id = img_info["image_id"]
            file_name = img_info["file_name"]
            self.instances.extend((
                                      image_id,
                                      file_name,
                                      anno["bbox"],
                                      anno["bbox_mode"],
                                      anno["segmentation"],
                                      anno["category_id"]
                                  ) for anno in img_info["annotations"])

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dataset_dicts = load_coco_json(
        r"/public/datasets/coco2017/annotations/instances_train.json",
        r"/public/datasets/coco2017/train2017",
        r"coco2017"
    )
    dir(dataset_dicts)
