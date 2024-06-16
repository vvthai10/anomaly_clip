import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
from tqdm import tqdm

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'BrainMRI':
        obj_list = ['brain']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    elif dataset_name == 'medical':
        obj_list = ['Brain_AD', 'Histopathology_AD', 'Liver_AD', 'Retina_OCT2017_AD', 'Retina_RESC_AD']
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
        img = Image.open(os.path.join(self.root, img_path))
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            try:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            except:
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')

        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask
        return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}


CLASS_NAMES = ['Brain_AD', 'Liver_AD', 'Retina_RESC_AD', 'Retina_OCT2017_AD', 'Chest_AD', 'Histopathology_AD'] #
CLASS_INDEX = {'Brain_AD':3, 'Liver_AD':2, 'Retina_RESC_AD':1, 'Retina_OCT2017_AD':-1, 'Chest_AD':-2, 'Histopathology_AD':-3} #

class DatasetMedical(data.Dataset):
    def __init__(self, root, batch_size, img_size, transform, target_transform, dataset_name, mode='test'):
        self.root = root
        self.img_size = img_size
        self.batch_size = batch_size
        self.transform = transform
        self.target_transform = target_transform
        self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)
        
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        self.meta_info = meta_info[mode]
        self.cls_names = list(self.meta_info.keys())

        self.data_all = self.load_data_folder()

    def __len__(self):
        return len(self.data_all)

    def __getitem__(self, index):
        x, y, mask, seg_idx = self.data_all[index]

        batch = len(x)
        batch_img = []
        for i in range(batch):
            img = Image.open(os.path.join(self.root, x[i]))
            img = self.transform(img) if self.transform is not None else img
            batch_img.append(img.unsqueeze(0))
        batch_img = torch.cat(batch_img)

        if seg_idx < 0:
            # return batch_img, y, torch.zeros([1, self.resize, self.resize]), seg_idx
            return {'img': batch_img, 'img_mask': torch.zeros([1, self.img_size, self.img_size]), 'cls_idx': seg_idx, 'anomaly': y,
                    'img_path': []}

        batch_mask = []
        for i in range(batch):
            if y[i] == 1:
                img_mask = np.array(Image.open(os.path.join(self.root, mask[i])).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            else:
                img_mask = Image.fromarray(np.zeros((self.img_size, self.img_size)), mode='L')

            img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
            batch_mask.append(img_mask.unsqueeze(0))

        batch_mask = torch.cat(batch_mask)

        return {'img': batch_img, 'img_mask': batch_mask,
                'cls_idx': seg_idx, 'anomaly': y,
                'img_path': []}

    def load_data_folder(self):
        
        data = []
        for cls_name in self.cls_names:
            random.shuffle(self.meta_info[cls_name])
        for cls_name in self.cls_names:
            # self.data_all.extend(meta_info[cls_name])
            if CLASS_INDEX[cls_name] < 0:
                for image_index in range(0, len(self.meta_info[cls_name]), self.batch_size):
                    file_path = []
                    img_label = []
                    for batch_count in range(0, self.batch_size):
                        if image_index + batch_count >= len(self.meta_info[cls_name]):
                            break
                        file_path.append(self.meta_info[cls_name][image_index + batch_count]['img_path'])
                        img_label.append(self.meta_info[cls_name][image_index + batch_count]['anomaly'])
                    data.append([file_path, img_label, None, CLASS_INDEX[cls_name]])
            else:
                for image_index in range(0, len(self.meta_info[cls_name]), self.batch_size):
                    file_path = []
                    img_label = []
                    gt_path = []
                    for batch_count in range(0, self.batch_size):
                        if image_index + batch_count >= len(self.meta_info[cls_name]):
                            break
                        single_file_path = self.meta_info[cls_name][image_index + batch_count]['img_path']
                        single_label = self.meta_info[cls_name][image_index + batch_count]['anomaly']

                        file_path.append(single_file_path)
                        img_label.append(single_label)

                        if single_label == 0:
                            gt_path.append(None)
                        else:
                            gt_path.append(single_file_path.replace('img', 'anomaly_mask'))
                    data.append([file_path, img_label, gt_path, CLASS_INDEX[cls_name]])
        random.shuffle(data)
        return data

    def shuffle_dataset(self):
        for cls_name in self.cls_names:
            random.shuffle(self.meta_info[cls_name])
        data = []
        for cls_name in self.cls_names:
            if CLASS_INDEX[cls_name] < 0:
                for image_index in range(0, len(self.meta_info[cls_name]), self.batch_size):
                    file_path = []
                    img_label = []
                    for batch_count in range(0, self.batch_size):
                        if image_index + batch_count >= len(self.meta_info[cls_name]):
                            break
                        file_path.append(self.meta_info[cls_name][image_index + batch_count]['img_path'])
                        img_label.append(self.meta_info[cls_name][image_index + batch_count]['anomaly'])
                    data.append([file_path, img_label, None, CLASS_INDEX[cls_name]])
            else:
                for image_index in range(0, len(self.meta_info[cls_name]), self.batch_size):
                    file_path = []
                    img_label = []
                    gt_path = []
                    for batch_count in range(0, self.batch_size):
                        if image_index + batch_count >= len(self.meta_info[cls_name]):
                            break
                        single_file_path = self.meta_info[cls_name][image_index + batch_count]['img_path']
                        single_label = self.meta_info[cls_name][image_index + batch_count]['anomaly']

                        file_path.append(single_file_path)
                        img_label.append(single_label)

                        if single_label == 0:
                            gt_path.append(None)
                        else:
                            gt_path.append(single_file_path.replace('img', 'anomaly_mask'))
                    data.append([file_path, img_label, gt_path, CLASS_INDEX[cls_name]])
        random.shuffle(data)
        self.data = data

if __name__ == '__main__':
    train_data = DatasetMedical(root="./data/medical", batch_size=6, transform=None,
                                target_transform=None, dataset_name="medical")
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    for items in tqdm(train_dataloader, position=0, leave=True):
        image = items['img'].to("cuda")
        label = items['anomaly']

        gt = items['img_mask'].squeeze().to("cuda")
        gt[gt > 0.5] = 1
        gt[gt <= 0.5] = 0