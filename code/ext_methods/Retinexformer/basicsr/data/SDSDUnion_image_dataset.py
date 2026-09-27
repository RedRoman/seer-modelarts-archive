"""SDSDUnionImage - Concatenation of SDSD-indoor + SDSD-outdoor for the Seer-protocol
   controlled comparison retrain. Two roots concatenated at the index level,
   otherwise identical to Dataset_SDSDImage."""
import os.path as osp
import torch
import torch.utils.data as data
import basicsr.data.util as util
import numpy as np


class Dataset_SDSDUnionImage(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.io_backend_opt = opt['io_backend']
        self.data_type = self.io_backend_opt['type']
        if self.data_type == 'lmdb':
            raise ValueError('LMDB not supported here.')
        self.data_info = {'path_LQ': [], 'path_GT': [],
                          'folder': [], 'idx': [], 'border': []}
        self.imgs_LQ, self.imgs_GT = {}, {}

        # opt['dataroot_gt'] / opt['dataroot_lq'] are LISTS in this union dataset
        gt_roots = opt['dataroot_gt']
        lq_roots = opt['dataroot_lq']
        testing_dirs_all = opt['testing_dir']  # list-of-lists, one per root
        assert isinstance(gt_roots, list) and isinstance(lq_roots, list)
        assert len(gt_roots) == len(lq_roots) == len(testing_dirs_all)

        for gt_root, lq_root, testing_dir_str in zip(gt_roots, lq_roots, testing_dirs_all):
            if testing_dir_str is not None:
                testing_dir = testing_dir_str.split(',')
            else:
                testing_dir = []
            print(f'[SDSDUnion] root_gt={gt_root} testing_dir={testing_dir}')

            subfolders_LQ = util.glob_file_list(lq_root)
            subfolders_GT = util.glob_file_list(gt_root)

            for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
                subfolder_name = osp.basename(subfolder_GT)
                if self.opt['phase'] == 'train':
                    if (subfolder_name in testing_dir):
                        continue
                    if (subfolder_name.split('_2')[0] in testing_dir):
                        continue
                else:
                    if not(subfolder_name in testing_dir) and not(subfolder_name.split('_2')[0] in testing_dir):
                        continue

                img_paths_LQ = util.glob_file_list(subfolder_LQ)
                img_paths_GT = util.glob_file_list(subfolder_GT)
                max_idx = len(img_paths_LQ)
                assert max_idx == len(img_paths_GT), 'LQ/GT count mismatch'
                # Namespace folder name with the root basename to keep uniqueness
                root_tag = osp.basename(osp.normpath(gt_root)).split('_')[0]  # 'indoor' or 'outdoor'
                unique_folder = f'{root_tag}__{subfolder_name}'
                self.data_info['path_LQ'].extend(img_paths_LQ)
                self.data_info['path_GT'].extend(img_paths_GT)
                self.data_info['folder'].extend([unique_folder] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append('{}/{}'.format(i, max_idx))
                border_l = [0] * max_idx
                for i in range(self.half_N_frames):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)
                if self.cache_data:
                    self.imgs_LQ[unique_folder] = img_paths_LQ
                    self.imgs_GT[unique_folder] = img_paths_GT

        print(f'[SDSDUnion] total samples: {len(self.data_info["path_LQ"])} in {len(self.imgs_LQ)} sequences.')

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx_max = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx_max[0]), int(idx_max[1])
        border = self.data_info['border'][index]

        img_LQ_path = self.imgs_LQ[folder][idx:idx + 1]
        img_GT_path = self.imgs_GT[folder][idx:idx + 1]

        img_LQ = util.read_img_seq2(img_LQ_path, self.opt['train_size'])[0]
        img_GT = util.read_img_seq2(img_GT_path, self.opt['train_size'])[0]

        if self.opt['phase'] == 'train':
            img_LQ_l = [img_LQ, img_GT]
            rlt = util.augment_torch(
                img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]; img_GT = rlt[1]

        return {
            'lq': img_LQ, 'gt': img_GT,
            'folder': folder, 'idx': self.data_info['idx'][index],
            'border': border,
            'lq_path': img_LQ_path[0], 'gt_path': img_GT_path[0]
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])
