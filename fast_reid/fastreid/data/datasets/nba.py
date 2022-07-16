# Custom NBA dataset

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class NBA(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = 'NBA'
    dataset_url = ''
    dataset_name = "NBA"

    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = self.dataset_dir

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.extra_gallery = False

        required_files = [
            self.data_dir,
            self.train_dir,
            # self.query_dir,
            # self.gallery_dir,
        ]

        self.check_before_run(required_files)

        train = lambda: self.process_dir(self.train_dir)
        query = lambda: self.process_dir(self.gallery_dir, is_train=False, default_camid=0)
        gallery = lambda: self.process_dir(self.gallery_dir, is_train=False) + \
                          (self.process_dir(self.extra_gallery_dir, is_train=False) if self.extra_gallery else [])
        super(NBA, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True, default_camid=1):

        img_paths = glob.glob(osp.join(dir_path, '*.png'))
        data = []
        for img_path in img_paths:
            player_int_id, *_ = img_path.split('/')[-1].split('-')
            player_int_id = int(player_int_id)
            camid = default_camid
            if is_train:
                player_int_id = self.dataset_name + "_" + str(player_int_id)
                camid = self.dataset_name + f"_{camid}"
            data.append((img_path, player_int_id, camid))

        return data
