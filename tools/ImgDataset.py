import os
import numpy as np
import glob
import torch.utils.data
from PIL import Image
import torch
from torchvision import transforms


class ModelNet_MultiView(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, 
                 num_views=20, total_num_views=20, num_classes=40):
        assert num_classes in [10, 40], '`num_classes` should be chosen in this list: [10, 40]'
        if num_classes == 40:
            self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                            'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                            'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                            'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                            'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        elif num_classes == 10:
            self.classnames=['bathtub', 'bed', 'chair', 'desk', 'dresser',
                             'monitor', 'night_stand', 'sofa', 'table', 'toilet']

        if not test_mode:
            self.root_dir = root_dir + '/*/train'
        else:
            self.root_dir = root_dir + '/*/test'
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        set_ = self.root_dir.split('/')[-1]  # --> train
        parent_dir = self.root_dir.rsplit('/',2)[0]  # --> data/modelnet40v2png_ori4
        self.filepaths = []
        for i in range(len(self.classnames)):
            # sorted 这个方法很巧妙
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            self.filepaths.extend(all_files)

        # NOTE `total_num_views` depends on the dataset where each 3D object corresponds maximum number of views
        #   `num_views` <= `total_num_views`, we can vary `num_views` to conduct ablation studies
        
        if not test_mode:
            # 训练模式，打乱顺序
            rand_idx = np.random.permutation(len(self.filepaths) // total_num_views)
        else:
            # 测试模式，不打乱顺序
            rand_idx = list(range(len(self.filepaths) // total_num_views))

        filepaths_shuffled = []
        for i in range(len(rand_idx)):
            idx = rand_idx[i]
            start = idx * total_num_views
            end = (idx+1) * total_num_views
            filepaths_interval = self.filepaths[start:end]
            # NOTE randomly select `num_view` views from `filepaths_interval`
            #   use `np.random.choice`, it is necessary to set `replace=False`
            selected_filepaths = np.random.choice(filepaths_interval, size=(num_views,), replace=False)
            filepaths_shuffled.extend(selected_filepaths)
        self.filepaths = filepaths_shuffled

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])


class ModelNet_SingleView(torch.utils.data.Dataset):
    def __init__(self, root_dir, scale_aug=False, rot_aug=False, test_mode=False, num_classes=40):
        assert num_classes in [10, 40], '`num_classes` should be chosen in this list: [10, 40]'
        if num_classes == 40:   # ModelNet40
            self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                            'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                            'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                            'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                            'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        elif num_classes == 10: # ModelNet10
            self.classnames=['bathtub', 'bed', 'chair', 'desk', 'dresser',
                             'monitor', 'night_stand', 'sofa', 'table', 'toilet']
                            
        if not test_mode:
            self.root_dir = root_dir + '/*/train'
        else:
            self.root_dir = root_dir + '/*/test'
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        set_ = self.root_dir.split('/')[-1]
        parent_dir = self.root_dir.rsplit('/',2)[0]
        self.filepaths = []
        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir+'/'+self.classnames[i]+'/'+set_+'/*.png'))
            self.filepaths.extend(all_files)

        # NOTE For single-view dataset, when Training, there is no need to shuffle `self.filepaths` manually,
        #       torch.utils.data.DataLoader(shuffle=True) helps us finish it.

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)


class RGBD_MultiView(torch.utils.data.Dataset):
    def __init__(self, root_dir='rgbd-dataset_eval', testinstances_file='rgbd-dataset_eval_testinstance_ids.txt', 
                 trial_id=1, num_views=12, total_num_views=20, test_mode=False, num_classes=51):
        '''
        NOTE 假设一个 batch 包含 m 个样本，对于MultiView，每个样本对应 num_views 张图像
                RGBD_MultiView 要保证这 num_views 张图像对应同一个物体实例
        '''

        # 搜集测试实例列表
        test_instances = []
        trial_tag = f'trial {trial_id}'
        flag = False
        with open(os.path.join(root_dir, testinstances_file)) as f:
            lines = f.readlines()
            for line in lines:
                if trial_tag in line.strip():
                    flag = True
                    continue

                if len(test_instances) < num_classes:
                    if flag == True:
                        test_instances.append(line.strip())
                else:
                    break
        self.num_views = num_views
        self.classnames = []
        self.filepaths = []
        for obj_class in os.listdir(root_dir):
            # 判断是否为目录，如果不是继续判断下一个
            if not os.path.isdir(os.path.join(root_dir, obj_class)):
                continue

            self.classnames.append(obj_class)
            work_dir = os.path.join(root_dir, obj_class)
            for obj_instance in os.listdir(work_dir):
                # 当前实例属于测试实例
                if obj_instance in test_instances:
                    # 数据集为测试模式，将测试实例加入文件路径
                    if test_mode:
                        inst_dir = os.path.join(work_dir, obj_instance)
                        # NOTE 这里从 inst_dir 抽出来的图像，一定来自同一个实例
                        #   需要设定选多少张图片
                        #       1. RotationNet, View-GCN 都说每个实例用了12张图，这样训练数据量够吗？我觉得不够，可以测试效果
                        #       2. 我选择 num_views 的最大整数倍的图片数量
                        images = glob.glob(f'{inst_dir}/*_crop.png')
                        # glob出来的images排序是随机的，我觉得还是不排序了，反正都是同一物体的实例
                        # images.sort()   # in-place operation
                        batch = len(images) // num_views
                        self.filepaths.extend(images[:batch*num_views])
                # 当前实例属于训练实例
                else:
                    # 数据集为训练模式
                    if not test_mode:
                        inst_dir = os.path.join(work_dir, obj_instance)
                        # NOTE 这里从 inst_dir 抽出来的图像，一定来自同一个实例
                        #   需要设定选多少张图片
                        #       1. RotationNet, View-GCN 都说每个实例用了12张图，这样训练数据量够吗？我觉得不够，可以测试效果
                        #       2. 我选择 num_views 的最大整数倍的图片数量
                        images = glob.glob(f'{inst_dir}/*_crop.png')
                        # glob出来的images排序是随机的，我觉得还是不排序了，反正都是同一物体的实例
                        # images.sort()   # in-place operation
                        batch = len(images) // num_views
                        self.filepaths.extend(images[:batch*num_views])
        # it is necessary to sort `self.classnames`, ensuring `classnames` in order in train/test/val 
        self.classnames = sorted(self.classnames)

        if not test_mode:
            # 训练模式，打乱顺序
            rand_idx = np.random.permutation(len(self.filepaths) // total_num_views)
        else:
            # 测试模式，不打乱顺序
            rand_idx = list(range(len(self.filepaths) // total_num_views))

        filepaths_shuffled = []
        for i in range(len(rand_idx)):
            idx = rand_idx[i]
            start = idx * total_num_views
            end = (idx+1) * total_num_views
            filepaths_interval = self.filepaths[start:end]

            # NOTE randomly select `num_view` views from `filepaths_interval`
            #   use `np.random.choice`, it is necessary to set `replace=False`
            selected_filepaths = np.random.choice(filepaths_interval, size=(num_views,), replace=False)
            filepaths_shuffled.extend(selected_filepaths)
        self.filepaths = filepaths_shuffled

        self.transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        path = self.filepaths[idx*self.num_views]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])


class RGBD_SingleView(torch.utils.data.Dataset):
    def __init__(self, root_dir='rgbd-dataset_eval', testinstances_file='rgbd-dataset_eval_testinstance_ids.txt', 
                 trial_id=1, test_mode=False, num_classes=51):
        
        # 搜集测试实例列表
        test_instances = []
        trial_tag = f'trial {trial_id}'
        flag = False
        with open(os.path.join(root_dir, testinstances_file)) as f:
            lines = f.readlines()
            for line in lines:
                if trial_tag in line.strip():
                    flag = True
                    continue

                if len(test_instances) < num_classes:
                    if flag == True:
                        test_instances.append(line.strip())
                else:
                    break

        self.classnames = []
        self.filepaths = []
        for obj_class in os.listdir(root_dir):
            # 判断是否为目录，如果不是继续判断下一个
            if not os.path.isdir(os.path.join(root_dir, obj_class)):
                continue

            self.classnames.append(obj_class)
            work_dir = os.path.join(root_dir, obj_class)
            for obj_instance in os.listdir(work_dir):
                # 当前实例属于测试实例
                if obj_instance in test_instances:
                    # 数据集为测试模式，将测试实例加入文件路径
                    if test_mode:
                        inst_dir = os.path.join(work_dir, obj_instance)
                        images = glob.glob(f'{inst_dir}/*_crop.png')
                        self.filepaths.extend(images)
                # 当前实例属于训练实例
                else:
                    # 数据集为训练模式
                    if not test_mode:
                        inst_dir = os.path.join(work_dir, obj_instance)
                        images = glob.glob(f'{inst_dir}/*_crop.png')
                        self.filepaths.extend(images)
        # it is necessary to sort `self.classnames`, ensuring `classnames` in order in train/test/val 
        self.classnames = sorted(self.classnames)

        self.transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        class_name = path.split('/')[-3]
        class_id = self.classnames.index(class_name)
        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)
    
    
class ShapeNetCore55_MultiView(torch.utils.data.Dataset):
    def __init__(self, root_dir='data/shrec17', label_file='train.csv',  
                 version='normal', num_views=20, total_num_views=20, num_classes=55):
        assert num_classes in [55,], '`num_classes` should be chosen in this list: [55,]'
        if num_classes == 55:
            self.classnames = list()
            self.shape2class = dict()
            with open(os.path.join(root_dir, label_file)) as fin:
                lines = fin.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue    # skip csv header line
                    shape_name, class_name, _ = line.strip().split(',')
                    if shape_name not in self.shape2class.keys():
                        self.shape2class[shape_name] = class_name
                    if class_name not in self.classnames:
                        self.classnames.append(class_name)
            # it is necessary to sort `self.classnames`, ensuring `classnames` in order in train/test/val 
            self.classnames = sorted(self.classnames)

        mode = os.path.splitext(label_file)[0]
        self.root_dir = root_dir
        # e.g, work_dir = 'data/shrec17/train_normal'
        self.work_dir = os.path.join(self.root_dir, f'{mode}_{version}')

        self.num_views = num_views
        self.filepaths = []

        # sorted 这个方法很巧妙
        all_files = sorted(glob.glob(f'{self.work_dir}/*.png'))
        self.filepaths.extend(all_files)

        # NOTE `total_num_views` depends on the dataset where each 3D object corresponds maximum number of views
        #   `num_views` <= `total_num_views`, we can vary `num_views` to conduct ablation studies
        
        if mode == 'train':
            # 训练模式，打乱顺序
            rand_idx = np.random.permutation(len(self.filepaths) // total_num_views)
        else:
            # 验证/测试模式，不打乱顺序
            rand_idx = list(range(len(self.filepaths) // total_num_views))

        filepaths_shuffled = []
        for i in range(len(rand_idx)):
            idx = rand_idx[i]
            start = idx * total_num_views
            end = (idx+1) * total_num_views
            filepaths_interval = self.filepaths[start:end]
            # NOTE randomly select `num_view` views from `filepaths_interval`
            #   use `np.random.choice`, it is necessary to set `replace=False`
            selected_filepaths = np.random.choice(filepaths_interval, size=(num_views,), replace=False)
            filepaths_shuffled.extend(selected_filepaths)
        self.filepaths = filepaths_shuffled

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return int(len(self.filepaths)/self.num_views)

    def __getitem__(self, idx):
        # e.g. 'data/shrec17/val_normal/019180_003.png'
        path = self.filepaths[idx*self.num_views]
        # e.g. '019180_003.png'
        img_name = path.split('/')[-1]
        # e.g. '019180'
        shape_name = img_name[:6]
        # e.g. '04256520'
        class_name = self.shape2class[shape_name]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        imgs = []
        for i in range(self.num_views):
            im = Image.open(self.filepaths[idx*self.num_views+i]).convert('RGB')
            if self.transform:
                im = self.transform(im)
            imgs.append(im)

        return (class_id, torch.stack(imgs), self.filepaths[idx*self.num_views:(idx+1)*self.num_views])


class ShapeNetCore55_SingleView(torch.utils.data.Dataset):
    def __init__(self, root_dir='data/shrec17', label_file='train.csv', version='normal', num_classes=55):
        assert num_classes in [55,], '`num_classes` should be chosen in this list: [55,]'
        if num_classes == 55:
            self.classnames = list()
            self.shape2class = dict()
            with open(os.path.join(root_dir, label_file)) as fin:
                lines = fin.readlines()
                for i, line in enumerate(lines):
                    if i == 0:
                        continue    # skip csv header line
                    shape_name, class_name, _ = line.strip().split(',')
                    if shape_name not in self.shape2class.keys():
                        self.shape2class[shape_name] = class_name
                    if class_name not in self.classnames:
                        self.classnames.append(class_name)
            # it is necessary to sort `self.classnames`, ensuring `classnames` in order in train/test/val 
            self.classnames = sorted(self.classnames)

        mode = os.path.splitext(label_file)[0]
        self.root_dir = root_dir
        # e.g, work_dir = 'data/shrec17/train_normal'
        self.work_dir = os.path.join(self.root_dir, f'{mode}_{version}')

        self.filepaths = []

        all_files = sorted(glob.glob(f'{self.work_dir}/*.png'))
        self.filepaths.extend(all_files)

        # NOTE For single-view dataset, when Training, there is no need to shuffle `self.filepaths` manually,
        #       torch.utils.data.DataLoader(shuffle=True) helps us finish it.

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # e.g. 'data/shrec17/val_normal/019180_003.png'
        path = self.filepaths[idx]
        # e.g. '019180_003.png'
        img_name = path.split('/')[-1]
        # e.g. '019180'
        shape_name = img_name[:6]
        # e.g. '04256520'
        class_name = self.shape2class[shape_name]
        class_id = self.classnames.index(class_name)

        # Use PIL instead
        im = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            im = self.transform(im)
        return (class_id, im, path)

