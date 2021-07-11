# Library Imports
from multiprocessing import Pool
import os.path as osp
import pandas as pd
import numpy as np

# MMDetection Imports
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.core.evaluation.mean_ap import get_cls_results
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmcv.utils import print_log
import mmcv


def calc_tpfpfn(det_bboxes, gt_bboxes, iou_thr=0.5):
    """Check if detected bboxes are true positive or false positive and if gt bboxes are false negative.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.

    Returns:
        float: (tp, fp, fn).
    """
    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    tp = 0
    fp = 0

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if num_gts == 0:
        fp = num_dets
        return tp, fp, 0

    ious: np.ndarray = bbox_overlaps(det_bboxes, gt_bboxes)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
        uncovered_ious = ious[i, gt_covered == 0]
        if len(uncovered_ious):
            iou_argmax = uncovered_ious.argmax()
            iou_max = uncovered_ious[iou_argmax]
            if iou_max >= iou_thr:
                gt_covered[[x[iou_argmax] for x in np.where(gt_covered == 0)]] = True
                tp += 1
            else:
                fp += 1
        else:
            fp += 1
    fn = (gt_covered == 0).sum()
    return tp, fp, fn


def kaggle_map(
    det_results, annotations, iou_thrs=(0.5, 0.6, 0.65, 0.7, 0.75), logger=None, n_jobs=4, by_sample=False
):
    """Evaluate kaggle mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        iou_thrs (list): IoU thresholds to be considered as matched.
            Default: (0.5, 0.55, 0.6, 0.65, 0.7, 0.75).
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        n_jobs (int): Processes used for computing TP, FP and FN.
            Default: 4.
        by_sample (bool): Return AP by sample.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)

    num_imgs = len(det_results)
    num_classes = len(det_results[0])  # positive class num

    pool = Pool(n_jobs)
    eval_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, _ = get_cls_results(det_results, annotations, i)
        # compute tp and fp for each image with multiple processes
        aps_by_thrs = []
        aps_by_sample = np.zeros(num_imgs)
        for iou_thr in iou_thrs:
            tpfpfn = pool.starmap(calc_tpfpfn, zip(cls_dets, cls_gts, [iou_thr for _ in range(num_imgs)]))
            iou_thr_aps = np.array([tp / (tp + fp + fn) for tp, fp, fn in tpfpfn])
            print_log(f'IOU THRESH: {iou_thr}: {iou_thr_aps}')
            if by_sample:
                aps_by_sample += iou_thr_aps
            aps_by_thrs.append(np.mean(iou_thr_aps))
        eval_results.append(
            {
                "num_gts": len(cls_gts),
                "num_dets": len(cls_dets),
                "ap": np.mean(aps_by_thrs),
                "ap_by_sample": None if not by_sample else aps_by_sample / len(iou_thrs),
            }
        )
    pool.close()

    aps = []
    for cls_result in eval_results:
        if cls_result["num_gts"] > 0:
            aps.append(cls_result["ap"])
    mean_ap = np.array(aps).mean().item() if aps else 0.0

    print_log(f"\nKaggle mAP: {mean_ap}", logger=logger)
    print_log(f'mean_ap, eval_results: {mean_ap, eval_results}', logger=logger)
    return mean_ap, eval_results

def calc_pseudo_confidence(sample_scores, pseudo_score_threshold):
    if len(sample_scores):
        return np.sum(sample_scores > pseudo_score_threshold) / len(sample_scores)
    else:
        return 0.0

@DATASETS.register_module()
class KaggleDataset(CocoDataset): 
    CLASSES = ('opacity',)
    def evaluate(self, results, logger=None, iou_thrs=(0.5), **kwargs):
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        mean_ap, _ = kaggle_map(results, annotations, iou_thrs=iou_thrs, logger=logger)
        return dict(mAP=mean_ap)

    def format_results(self, results, output_path=None, **kwargs):
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )
        prediction_results = []
        for idx in range(len(self)):
            wheat_bboxes = results[idx][0]

            prediction_strs = []
            for bbox in wheat_bboxes:
                x, y, w, h = self.xyxy2xywh(bbox)
                prediction_strs.append(f"{bbox[4]:.4f} {x} {y} {w} {h}")
            filename = self.data_infos[idx]["filename"]
            image_id = osp.splitext(osp.basename(filename))[0]
            prediction_results.append({"image_id": image_id, "PredictionString": " ".join(prediction_strs)})
        predictions = pd.DataFrame(prediction_results)
        if output_path is not None:
            predictions.to_csv(output_path, index=False)
        return predictions

    def evaluate_by_sample(self, results, output_path, logger=None, iou_thrs=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75)):
        annotations = [self.get_ann_info(i) for i in range(len(self))]
        _, eval_results = kaggle_map(results, annotations, iou_thrs=iou_thrs, logger=logger, by_sample=True)
        output_annotations = self.coco.dataset["annotations"]
        output_images = []

        for idx in range(len(self)):
            wheat_bboxes = results[idx][0]
            data_info = self.data_infos[idx]
            data_info["ap"] = eval_results[0]["ap_by_sample"][idx]
            output_images.append(data_info)
            for bbox in wheat_bboxes:
                x, y, w, h = map(float, self.xyxy2xywh(bbox))
                output_annotations.append(
                    {
                        "segmentation": "",
                        "area": w * h,
                        "image_id": data_info["id"],
                        "category_id": 2,
                        "bbox": [x, y, w, h],
                        "iscrowd": 0,
                        "score": float(bbox[-1]),
                    }
                )
        for i, ann in enumerate(output_annotations):
            ann["id"] = i
        outputs = {
            "annotations": output_annotations,
            "images": output_images,
            "categories": [
                {"supercategory": "wheat", "name": "gt", "id": 1},
                {"supercategory": "wheat", "name": "predict", "id": 2},
            ],
        }
        mmcv.dump(outputs, output_path)
        return outputs

    def pseudo_results(self, results, output_path=None, pseudo_score_threshold=0.8, pseudo_confidence_threshold=0.65):
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )
        pseudo_annotations = []
        pseudo_images = []
        for idx in range(len(self)):
            wheat_bboxes = results[idx][0]
            scores = np.array([bbox[-1] for bbox in wheat_bboxes])
            confidence = calc_pseudo_confidence(scores, pseudo_score_threshold=pseudo_score_threshold)
            if confidence >= pseudo_confidence_threshold:
                data_info = self.data_infos[idx]
                data_info["confidence"] = confidence
                pseudo_images.append(data_info)
                for bbox in wheat_bboxes:
                    x, y, w, h = self.xyxy2xywh(bbox)
                    pseudo_annotations.append(
                        {
                            "segmentation": "",
                            "area": w * h,
                            "image_id": data_info["id"],
                            "category_id": 1,
                            "bbox": [x, y, w, h],
                            "iscrowd": 0,
                        }
                    )
        for i, ann in enumerate(pseudo_annotations):
            ann["id"] = i
        print(len(pseudo_images))
        mmcv.dump(
            {
                "annotations": pseudo_annotations,
                "images": pseudo_images,
                "categories": [{"supercategory": "wheat", "name": "wheat", "id": 1}],
            },
            output_path,
        )
