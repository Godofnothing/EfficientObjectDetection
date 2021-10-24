import numpy as np

from typing import List


def convert_predictions(per_class_predictions: List[np.ndarray], conf_thr=0.5):
    # create list of bboxes and labels 
    bboxes = [np.zeros((0, 4)) for _ in per_class_predictions]
    labels = []

    for class_idx, class_predictions in enumerate(per_class_predictions):
        if len(class_predictions) > 0:
            conf_mask = class_predictions[:, 4] > conf_thr
            if conf_mask.sum() > 0:
                bboxes.append(class_predictions[conf_mask, :4])
                labels.extend([class_idx for _ in range(conf_mask.sum())])

    bboxes = np.concatenate(bboxes, axis=0)
    labels = np.array(labels, dtype=np.int8)

    return bboxes, labels
        