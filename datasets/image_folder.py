import os
import cv2

from torch.utils.data import Dataset
from torchvision import transforms

class ImageFolder(Dataset):
    '''
    Simple Dataset for just loading the images
    '''

    def __init__(
        self, 
        image_dir : str,
        image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(
                [123.675/255, 116.28/255, 103.53/255], 
                [58.395/255, 57.12/255, 57.375/255]
            )
        ])
    ):
        '''
        Args:
            image_dir: str - path to directory with images
            image_transforms: transforms  preprocssing pipeline for image
        '''
        super().__init__()
        self.image_dir = image_dir
        self.image_filenames = os.listdir(image_dir)
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.image_filenames)

    def get_raw_image(self, idx):
        return cv2.imread(f"{self.image_dir}/{self.image_filenames[idx]}")

    def __getitem__(self, idx):
        image = cv2.imread(f"{self.image_dir}/{self.image_filenames[idx]}")
        image = self.image_transforms(image)
        return image
