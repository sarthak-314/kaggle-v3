import numpy as np
import random

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import Pad

@PIPELINES.register_module()
class RandomRotate90(object):
    def __init__(self, rotate_ratio=None):
        self.rotate_ratio = rotate_ratio
        if rotate_ratio is not None:
            assert 0 <= rotate_ratio <= 1

    def bbox_rot90(self, bboxes, img_shape, factor):
        assert bboxes.shape[-1] % 4 == 0
        h, w = img_shape[:2]
        rotated = bboxes.copy()
        if factor == 1:
            rotated[..., 0] = bboxes[..., 1]
            rotated[..., 1] = w - bboxes[..., 2]
            rotated[..., 2] = bboxes[..., 3]
            rotated[..., 3] = w - bboxes[..., 0]
        elif factor == 2:
            rotated[..., 0] = w - bboxes[..., 2]
            rotated[..., 1] = h - bboxes[..., 3]
            rotated[..., 2] = w - bboxes[..., 0]
            rotated[..., 3] = h - bboxes[..., 1]
        elif factor == 3:
            rotated[..., 0] = h - bboxes[..., 3]
            rotated[..., 1] = bboxes[..., 0]
            rotated[..., 2] = h - bboxes[..., 1]
            rotated[..., 3] = bboxes[..., 2]
        return rotated

    def __call__(self, results):
        if "rotate" not in results:
            rotate = True if np.random.rand() < self.rotate_ratio else False
            results["rotate"] = rotate
        if "rotate_factor" not in results:
            rotate_factor = random.randint(0, 3)
            results["rotate_factor"] = rotate_factor
        if results["rotate"]:
            # rotate image
            for key in results.get("img_fields", ["img"]):
                results[key] = np.ascontiguousarray(np.rot90(results[key], results["rotate_factor"]))
            results["img_shape"] = results["img"].shape
            # rotate bboxes
            for key in results.get("bbox_fields", []):
                results[key] = self.bbox_rot90(results[key], results["img_shape"], results["rotate_factor"])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f"(factor={self.factor})"


class BufferTransform(object):
    def __init__(self, min_buffer_size, p=0.5):
        self.p = p
        self.min_buffer_size = min_buffer_size
        self.buffer = []

    def apply(self, results):
        raise NotImplementedError

    def __call__(self, results):
        if len(self.buffer) < self.min_buffer_size:
            self.buffer.append(results.copy())
            return None
        if np.random.rand() <= self.p and len(self.buffer) >= self.min_buffer_size:
            random.shuffle(self.buffer)
            return self.apply(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nmin_buffer_size={self.min_buffer_size}),\n"
        repr_str += f"(\nratio={self.p})"
        return repr_str


@PIPELINES.register_module()
class Mosaic(BufferTransform):
    """
    Based on https://github.com/dereyly/mmdet_sota
    """

    def __init__(self, min_buffer_size=4, p=0.5, pad_val=0):
        assert min_buffer_size >= 4, "Buffer size for mosaic should be at least 4!"
        super(Mosaic, self).__init__(min_buffer_size=min_buffer_size, p=p)
        self.pad_val = pad_val

    def apply(self, results):
        # take four images
        a = self.buffer.pop()
        b = self.buffer.pop()
        c = self.buffer.pop()
        d = self.buffer.pop()
        # get max shape
        max_h = max(a["img"].shape[0], b["img"].shape[0], c["img"].shape[0], d["img"].shape[0])
        max_w = max(a["img"].shape[1], b["img"].shape[1], c["img"].shape[1], d["img"].shape[1])

        # cropping pipe
        padder = Pad(size=(max_h, max_w), pad_val=self.pad_val)

        # crop
        a, b, c, d = padder(a), padder(b), padder(c), padder(d)

        # check if cropping returns None => see above in the definition of RandomCrop
        if not a or not b or not c or not d:
            return results

        # offset bboxes in stacked image
        def offset_bbox(res_dict, x_offset, y_offset, keys=("gt_bboxes", "gt_bboxes_ignore")):
            for k in keys:
                if k in res_dict and res_dict[k].size > 0:
                    res_dict[k][:, 0::2] += x_offset
                    res_dict[k][:, 1::2] += y_offset
            return res_dict

        b = offset_bbox(b, max_w, 0)
        c = offset_bbox(c, 0, max_h)
        d = offset_bbox(d, max_w, max_h)

        # collect all the data into result
        top = np.concatenate([a["img"], b["img"]], axis=1)
        bottom = np.concatenate([c["img"], d["img"]], axis=1)
        results["img"] = np.concatenate([top, bottom], axis=0)
        results["img_shape"] = (max_h * 2, max_w * 2)

        for key in ["gt_labels", "gt_bboxes", "gt_labels_ignore", "gt_bboxes_ignore"]:
            if key in results:
                results[key] = np.concatenate([a[key], b[key], c[key], d[key]], axis=0)
        return results

    def __repr__(self):
        repr_str = self.__repr__()
        repr_str += f"(\npad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class Mixup(BufferTransform):
    def __init__(self, min_buffer_size=2, p=0.5, pad_val=0):
        assert min_buffer_size >= 2, "Buffer size for mosaic should be at least 2!"
        super(Mixup, self).__init__(min_buffer_size=min_buffer_size, p=p)
        self.pad_val = pad_val

    def apply(self, results):
        # take four images
        a = self.buffer.pop()
        b = self.buffer.pop()

        # get min shape
        max_h = max(a["img"].shape[0], b["img"].shape[0])
        max_w = max(a["img"].shape[1], b["img"].shape[1])

        # cropping pipe
        padder = Pad(size=(max_h, max_w), pad_val=self.pad_val)

        # crop
        a, b = padder(a), padder(b)

        # check if cropping returns None => see above in the definition of RandomCrop
        if not a or not b:
            return results

        # collect all the data into result
        results["img"] = ((a["img"].astype(np.float32) + b["img"].astype(np.float32)) / 2).astype(a["img"].dtype)
        results["img_shape"] = (max_h, max_w)

        for key in ["gt_labels", "gt_bboxes", "gt_labels_ignore", "gt_bboxes_ignore"]:
            if key in results:
                results[key] = np.concatenate([a[key], b[key]], axis=0)
        return results



img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu_train_transforms = [
    dict(type="RandomRotate90", p=1.0),
    dict(
        type="OneOf",
        transforms=[
            dict(type="HueSaturationValue", hue_shift_limit=10, sat_shift_limit=35, val_shift_limit=25),
            dict(type="RandomGamma"),
            dict(type="CLAHE"),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="RandomBrightnessContrast", brightness_limit=0.25, contrast_limit=0.25),
            dict(type="RGBShift", r_shift_limit=15, g_shift_limit=15, b_shift_limit=15),
        ],
        p=0.5,
    ),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur"),
            dict(type="MotionBlur"),
            dict(type="GaussNoise"),
            dict(type="ImageCompression", quality_lower=75),
        ],
        p=0.4,
    ),
    dict(
        type="CoarseDropout",
        max_holes=30,
        max_height=30,
        max_width=30,
        min_holes=5,
        min_height=10,
        min_width=10,
        fill_value=img_norm_cfg["mean"][::-1],
        p=0.4,
    ),
    dict(
        type="ModifiedShiftScaleRotate",
        shift_limit=0.3,
        rotate_limit=5,
        scale_limit=(-0.3, 0.75),
        border_mode=0,
        value=img_norm_cfg["mean"][::-1],
    ),
    dict(type="RandomBBoxesSafeCrop", num_rate=(0.5, 1.0), erosion_rate=0.2),
]
def get_train_pipeline(img_size): 
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="Resize", img_scale=(img_size, img_size), keep_ratio=False),
        dict(type="Mosaic", p=0.25, min_buffer_size=4, pad_val=img_norm_cfg["mean"][::-1]),
        dict(
            type="Albumentations",
            transforms=albu_train_transforms,
            keymap=dict(img="image", gt_masks="masks", gt_bboxes="bboxes"),
            update_pad_shape=False,
            skip_img_without_anno=True,
            bbox_params=dict(type="BboxParams", format="pascal_voc", label_fields=["labels"]),
            min_visibility=0.3,
            min_size=4,
            max_aspect_ratio=15,
        ),
        dict(type="Mixup", p=0.25, min_buffer_size=2, pad_val=img_norm_cfg["mean"][::-1]),
        dict(type="RandomFlip", flip_ratio=0.5),
        dict(type="Normalize", **img_norm_cfg),
        dict(type="Pad", size_divisor=32),
        dict(type="DefaultFormatBundle"),
        dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
    ]
    return train_pipeline
