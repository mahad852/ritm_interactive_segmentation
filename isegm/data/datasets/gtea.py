from pathlib import Path

import cv2
import os

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.data_utils import get_instances_mask
from PIL import Image

class GTEADataset(ISDataset):
    def __init__(self, dataset_path,
                 **kwargs):
        super(GTEADataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)

        self.images_dir = os.path.join(self.dataset_path, 'Images')
        self.masks_dir = os.path.join(self.dataset_path, 'Masks')

        self.dataset_samples = []
        self.mask_paths = []

        for image_name in os.listdir(self.images_dir):
            self.dataset_samples.append(os.path.join(self.images_dir, image_name))
            self.mask_paths.append(os.path.join(self.masks_dir, image_name.split('.')[0] + '.png'))

    def get_sample(self, index) -> DSample:
        image = cv2.imread(self.dataset_samples[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        instances_mask = get_instances_mask(Image.open(self.mask_paths[index]))
        
        return DSample(image, instances_mask, ignore_ids=[0], sample_id=index)