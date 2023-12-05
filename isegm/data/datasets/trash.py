from pathlib import Path

import cv2
import os

from isegm.data.base import ISDataset
from isegm.data.sample import DSample
import numpy as np

from pycocotools.coco import COCO

class TrashDataset(ISDataset):
    def __init__(self, dataset_path,
                 **kwargs):
        super(TrashDataset, self).__init__(**kwargs)

        self.dataset_path = Path(dataset_path)
        
        self.coco = COCO(os.path.join(self.dataset_path, 'instances_val_trashcan.json'))
        self.images_map = self.get_image_map(self.coco)
        self.dataset_samples = [id for id in self.images_map.keys()]

    def get_image_map(self, coco: COCO):
        annsIds = coco.getAnnIds()
        images_map = {}
        
        for ann in annsIds:
            img_file_name = coco.loadImgs(coco.loadAnns(ann)[0]['image_id'])[0]['file_name']
            point = [coco.loadAnns(ann)[0]['bbox'][1] + int(coco.loadAnns(ann)[0]['bbox'][3]/2), coco.loadAnns(ann)[0]['bbox'][0] + int(coco.loadAnns(ann)[0]['bbox'][2]/2)]
            if img_file_name not in images_map:
                images_map[img_file_name] = []
            images_map[img_file_name].append({'ann' : ann, 'point' : point})
        
        return images_map

    def get_mask(self, coco: COCO, ann_id):
        return coco.annToMask(coco.loadAnns(ann_id)[0])

    def get_sample(self, index) -> DSample:
        image_filename = self.dataset_samples[index]
        image_path = os.path.join(self.dataset_path, 'val', image_filename)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks = [self.get_mask(self.coco, ann_obj['ann']) for ann_obj in self.images_map[image_filename]]
        
        instances_mask = np.zeros(image.shape[:-1])
        
        ids = [0]
        for i, mask in enumerate(masks):
            instances_mask += (i + 1) * mask
            ids.append(i + 1)

        return DSample(image, instances_mask, objects_ids=ids, ignore_ids=[0], sample_id=index)