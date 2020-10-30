import os
import logging
from pathlib import Path
import contextlib
import io

import torch
import torch.utils.data as data
import pycocotools.mask as mask_util
import pytorch_lightning as pl

# temporary imports
from utils import Timer
from structures.boxes import BoxMode

from tqdm import tqdm

logger = logging.getLogger(__name__)


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

    for (img_dict, anno_dict_list) in tqdm(imgs_anns, desc="processing images"):
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


class COCODataset(object):
    """

    """
    pass


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dataset_dicts = load_coco_json(
        r"/public/datasets/coco2017/annotations/person_keypoints_train2017.json",
        r"/public/datasets/coco2017/train2017",
        r"coco2017"
    )
    dir(dataset_dicts)
