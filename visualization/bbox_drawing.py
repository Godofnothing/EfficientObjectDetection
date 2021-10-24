import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

import torch
from utils.coco import ID_TO_CLASS, ID_TO_CONT_ID


class DetectionVisualizer:

    def __init__(
        self, 
        num_classes=80, 
        id_to_class = ID_TO_CLASS, 
        id_to_cont_id = ID_TO_CONT_ID,
        figsize=(8, 8)
    ):
        self.num_classes = num_classes
        self.figsize = figsize

        self.id_to_cont_id = id_to_cont_id if id_to_cont_id else {i : i for i in range(num_classes)}
        self.cont_id_to_id = {cont_id : id for id, cont_id in id_to_cont_id.items()}
        self.id_to_class = id_to_class if id_to_class else self.id_to_cont_id

        self.colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(self.num_classes)]        

    def draw_patches_with_bboxes(
        self, 
        image, 
        bboxes, 
        labels, 
        map_to_cond_id=False,
        output=''
    ):
        fig, ax = plt.subplots(figsize=self.figsize)
                                              
        ax.imshow(image[..., ::-1])
        ax.axis('off');

        for bbox, label in zip(bboxes, labels):
            if map_to_cond_id:
                cont_label = self.id_to_cont_id[label]
            else:
                cont_label = label
                label = self.cont_id_to_id[cont_label]
                
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            bbox_rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=self.colors[cont_label], facecolor='none')        
            ax.add_patch(bbox_rect)

            # add text
            ax.text(
                x1, y1, self.id_to_class[label], 
                fontsize=16, fontfamily='serif', bbox=dict(facecolor=self.colors[cont_label])
            )

        if output:
            plt.savefig(output)

def draw_image_with_bboxes(
    image, 
    bboxes, 
    labels,
    ax=None,
    colors=None, 
    normalize_image=True,
    num_colors=91,
    figsize=(8, 8)):

    if not colors:
        colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(num_colors)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if normalize_image:
        image = image - image.min()
        assert image.max() > 0.0
        image = image / image.max()
    # if channels first
    if image.shape[0] == 3:
        ax.imshow(image.transpose(1, 2, 0))
    else:
        ax.imshow(image)

    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        bbox_rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=colors[label], facecolor='none')
        ax.add_patch(bbox_rect)

        # add text
        ax.text(
            x1, y1, label, 
            fontsize=16, fontfamily='serif', bbox=dict(facecolor=colors[label])
        )
    plt.tight_layout()
