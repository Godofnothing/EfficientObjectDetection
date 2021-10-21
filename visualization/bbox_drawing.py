import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
import torch


def draw_patches_with_bboxes(
    patches, 
    patches_bboxes, 
    labels, 
    colors=None, 
    num_colors=91,
    figsize=(8, 8)
):
    num_splits = int(np.sqrt(len(patches)))
    fig, ax = plt.subplots(nrows=num_splits, ncols=num_splits, figsize=figsize)

    if not colors:
        colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(num_colors)]

    for idx, (lr_patch, lr_patch_bboxes, patch_labels) in enumerate(zip(patches, patches_bboxes, labels)):
        i, j = idx // num_splits, idx % num_splits                                         
        ax[i, j].imshow(lr_patch[..., ::-1])
        ax[i, j].axis('off');
        
        for bbox, label in zip(lr_patch_bboxes, patch_labels):
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            bbox_rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=colors[label], facecolor='none')        
            ax[i, j].add_patch(bbox_rect)
            
            # add text
            ax[i, j].text(
                x1, y1, label, 
                fontsize=16, fontfamily='serif', bbox=dict(facecolor=colors[label])
            )
        
    plt.tight_layout()

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
