import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle


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