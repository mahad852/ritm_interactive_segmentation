from pathlib import Path

import cv2
import os

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import numpy as np

class WoodscapeDataset(ISDataset):
    def __init__(self, dataset_path,
                 **kwargs):
        super(WoodscapeDataset, self).__init__(**kwargs)
        
        self.val_dir = os.path.join(dataset_path, 'rgb_images')
        self.instances_dir = os.path.join(dataset_path, 'instance_annotations')

        self.image_map = {}
        self.dataset_samples = []

        for img_fname in os.listdir(self.val_dir):
            self.image_map[img_fname] = os.path.join(self.instances_dir, img_fname.split('.')[0] + '.json')
            self.dataset_samples.append(img_fname)

    def create_mask_from_polygon(self, segmentation, image_shape):
        # Parse the coordinates into a format suitable for OpenCV
        polygon = np.array(segmentation, np.int32).reshape((-1, 1, 2))
        
        # Create a blank mask with the same dimensions as the image
        mask = np.zeros(image_shape, np.uint8)
        
        # Draw and fill the polygon on the mask
        cv2.fillPoly(mask, [polygon], 1)
        
        return mask
    
    def get_sample(self, index) -> DSample:
        
        img_fname = self.dataset_samples[index]
        image_path = os.path.join(self.val_dir, img_fname)

        with open(self.image_map[img_fname]) as f:
            annotations = json.load(f)

        image_path = os.path.join(self.dataset_path, self.dataset_samples[index])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotations = annotations[img_fname.split('.')[0] + '.json']['annotation']
        
        instances_mask = np.zeros(image.shape[:-1])
        ids = []

        for ann_i, ann_obj in enumerate(annotations):
            polygon = ann_obj['segmentation']
            mask_gt = self.create_mask_from_polygon(polygon, image.shape[:-1])
            ids.append(ann_i + 1)

            instances_mask += mask_gt * (ann_i + 1)

        return DSample(image, instances_mask, objects_ids=ids, ignore_ids=[0], sample_id=index)