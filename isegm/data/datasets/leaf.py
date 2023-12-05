from pathlib import Path

import cv2
import os

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
from isegm.utils.data_utils import get_instances_mask
from PIL import Image

class LeafDataset(ISDataset):
    def __init__(self, dataset_path,
                 **kwargs):
        super(LeafDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_samples = [self.get_file_name(i) + '_rgb.png' for i in range(1, 121)]
    
    def get_file_name(self, image_num):
        prefix = 'ara2012_plant'
        num_str = '0' * (3 - len(str(image_num))) + str(image_num)
        
        return prefix + num_str

    def get_sample(self, index) -> DSample:

        image_path = os.path.join(self.dataset_path, self.dataset_samples[index])
        mask_path = os.path.join(self.dataset_path, self.get_file_name(index) + '_label.png')

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        instances_mask = get_instances_mask(Image.open(mask_path))

        return DSample(image, instances_mask, ignore_ids=[0], sample_id=index)