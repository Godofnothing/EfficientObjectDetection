import numpy as np

from typing import List


def convert_predictions(per_class_predictions: List[List[np.ndarray]]):
    # create list of bboxes and labels for each patch
    patches_bboxes = [[np.zeros((0, 4))] for _ in per_class_predictions]
    patches_labels = [[] for _ in per_class_predictions]

    for i, image_pcp in enumerate(per_class_predictions):
        for class_idx, class_predictions in enumerate(image_pcp):
            if len(class_predictions) > 0:
                patches_bboxes[i].append(class_predictions[:, :4])
                patches_labels[i].append(class_idx)

        patches_bboxes[i] = np.concatenate(patches_bboxes[i], axis=0)
        patches_labels[i] = np.array(patches_labels[i], dtype=np.int8)

    return patches_bboxes, patches_labels
        