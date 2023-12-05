from pathlib import Path

import cv2
import os

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import numpy as np
from pycocotools import mask as pycoco_mask
import json

class IShapeDataset(ISDataset):
    def __init__(self, dataset_path,
                 **kwargs):
        super(IShapeDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        self.dataset_samples = []
        self.images = {}

        for sub_dirs in os.listdir(self.dataset_path):
            if not os.path.isdir(os.path.join(self.dataset_path, sub_dirs)):
                continue

            val_dir = os.path.join(self.dataset_path, sub_dirs, 'val')
            annotations_obj = {}

            with open(os.path.join(val_dir, 'coco_format-mask_encoding=rle-instances_2017.json')) as f:
                annotations_obj = json.load(f)
            

            for img_obj in annotations_obj['images']:
                img_id = sub_dirs + img_obj['id']
                self.images[img_id] = {}
                self.images[img_id]['path'] = os.path.join(val_dir, 'image', img_obj['file_name'])
                self.images[img_id]['seg'] = []
                self.dataset_samples.append(img_id)

            for ann_obj in annotations_obj['annotations']:                
                img_id = sub_dirs + ann_obj['image_id']
                self.images[img_id]['seg'].append(pycoco_mask.decode(ann_obj['segmentation']))


    def get_sample(self, index) -> DSample:
        image_id = self.dataset_samples[index]

        image_path = os.path.join(self.images[image_id]['path'])

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        instances_mask = np.zeros(image.shape[0], image.shape[1])
        
        ids = []
        for i, mask in enumerate(self.images[image_id]['seg']):
            instances_mask += mask * (i + 1)
            ids.append(i + 1)

        return DSample(image, instances_mask, objects_ids=ids, ignore_ids=[0], sample_id=index)