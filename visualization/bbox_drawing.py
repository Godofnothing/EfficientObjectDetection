import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle


class DetectionVisualizer:

    def __init__(self, num_classes, class_names=None, figsize=(8, 8)):
        self.num_classes = num_classes
        self.figsize = figsize
        if not class_names:
            self.class_names = np.arange(self.num_classes)

        self.colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(self.num_classes)]        

    def draw_patches_with_bboxes(
        self, 
        patches, 
        patches_bboxes, 
        labels, 
        output=''
    ):
        num_splits = int(np.sqrt(len(patches)))
        fig, ax = plt.subplots(nrows=num_splits, ncols=num_splits, figsize=self.figsize)

        for idx, (lr_patch, lr_patch_bboxes, patch_labels) in enumerate(zip(patches, patches_bboxes, labels)):
            i, j = idx // num_splits, idx % num_splits                                         
            ax[i, j].imshow(lr_patch[..., ::-1])
            ax[i, j].axis('off');
            
            for bbox, label in zip(lr_patch_bboxes, patch_labels):
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1
                bbox_rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor=self.colors[label], facecolor='none')        
                ax[i, j].add_patch(bbox_rect)
                
                # add text
                ax[i, j].text(
                    x1, y1, self.class_names[label], 
                    fontsize=16, fontfamily='serif', bbox=dict(facecolor=self.colors[label])
                )
            
        plt.tight_layout()

        if output:
            plt.savefig(output)