'''
1. 抽取测试集object特征
2. 每个object与其他物体计算相似度分数，形成排序列表
3. 用shrec17提供的代码评测检索效果
'''

import os, sys
import numpy as np
from scipy.special import softmax

import torch
from torch.utils.data import DataLoader

BASE_DIR = '/your/absolute/path/to/view-GCN-DDP'
sys.path.append(BASE_DIR)
from model.view_gcn import SVCNN, view_GCN
from tools.ImgDataset import ShapeNetCore55_MultiView


# e.g. `val.csv`
label_file = sys.argv[1]
mode = os.path.splitext(label_file)[0]

# --- step 1: get object feature representation
# define model

num_obj_classes = 55
num_views = 20

# e.g., `alexnet`
model_name = sys.argv[2]
# e.g., 0
rank = int(sys.argv[3])
# e.g., SH17-V20-ViewGCN-RN18-2
exp_name = sys.argv[4]

sv_classifier = SVCNN('svcnn', nclasses=num_obj_classes, pretraining=True, cnn_name=model_name).to(rank)
model = view_GCN('viewgcn', sv_classifier, nclasses=num_obj_classes, cnn_name=model_name, num_views=num_views).to(rank)
model_weights = f'../runs/RET/viewgcn_shrec17/{exp_name}/weights/mv_model_best.pth'
map_location = torch.device('cuda:%d' % rank)
state_dict = torch.load(model_weights, map_location=map_location)
model.load_state_dict(state_dict, strict=True)

# define multi-view image dataset
total_num_views = 20
mv_test_set = ShapeNetCore55_MultiView(root_dir='../data/shrec17', label_file=label_file, 
    version='normal', num_views=num_views, total_num_views=total_num_views, num_classes=num_obj_classes)
num_test_objects = len(mv_test_set)
print('ShapeNet Core55 num_test_objects:', num_test_objects)

# load multi-view image dataset
# e.g., 132
mv_samples_per_gpu = int(sys.argv[5])
num_workers = 1
mv_test_loader = DataLoader(mv_test_set, batch_size=mv_samples_per_gpu, 
                            shuffle=False, num_workers=num_workers, pin_memory=True,)

# forward pass to get shape `category distribution`
with torch.no_grad():
    pred_logits = []
    pred_class_ids = []
    for data in mv_test_loader:
        B, V, C, H, W = data[1].size()
        in_data = data[1].view(-1, C, H, W).to(rank)
        
        # pred: [batch, num_obj_classes]
        logits = model(in_data)[0].cpu().numpy()
        # it is necessary to normalize logits using `scipy.special.softmax`
        logits = softmax(logits, axis=1)
        class_ids = logits.argmax(axis=1)

        pred_logits.extend(logits)
        pred_class_ids.extend(class_ids)

pred_logits = np.array(pred_logits)
pred_class_ids = np.array(pred_class_ids)

print('pred_logits.shape:', pred_logits.shape)
print('pred_class_ids.shape:', pred_class_ids.shape)

# --- step 2: compute rank list for each object
# 相似度计算方式：
#   1. 物体类别分布做排序依据，把同类物体返回，再按照概率大小排序
#        理解了 RotationNet 的排序过程，仿照它实现自己的
#   2. 物体特征表示计算相似度，形成长为N的rank list

from  os.path import basename, join, exists

shape_names = []
filepaths = np.array(mv_test_set.filepaths)[::num_views] # take one view every `num_views`
print('len(filepaths):', len(filepaths))

# NOTE 生成 test.csv 和 fp_test.txt
with open(label_file, 'w') as fout1, open(f'fp_{mode}.txt', 'w') as fout2:
    fout1.write('id,synsetId,subSynsetId,modelId,split\n')
    for fp in filepaths:
        filename = basename(fp)
        shape_name = filename[:6]
        shape_names.append(shape_name)

        class_name = mv_test_set.shape2class[shape_name]
        class_id = mv_test_set.classnames.index(class_name)

        fout1.write(f'{shape_name},{class_id},{class_id},{shape_name},{mode}\n')
        fout2.write(f'{fp}\n')

shape_names = np.array(shape_names)
num_objects = len(pred_logits)

method_name = sys.argv[6]
savedir = join('evaluator', method_name, f'{mode}_normal')
if not exists(savedir):
    os.mkdir(savedir)

for idx in range(num_objects):
    filename = join(savedir, shape_names[idx])
    with open(filename, 'w') as fout:   # viewformer/test_normal/000009
        scores_column = pred_logits[:, pred_class_ids[idx]].copy()
        scores_column[idx] = float('inf')
        
        # pick up sample ids in the same class
        ids = [i for i, class_id in enumerate(pred_class_ids) if class_id == pred_class_ids[idx]]
        scores_column_ = scores_column[ids]  # scores_column_ 的长度不一定 >=1000，往往是 <1000
        shape_names_ = shape_names[ids]

        # NOTE np.argsort with `asc` order，then it converts to `desc` order with [::-1] 
        target_ids = np.argsort(scores_column_)[::-1]
        if len(target_ids) > 1000:
            target_ids = target_ids[:1000]
        for i in target_ids:
            mesh_name = shape_names_[i]
            distance = '{:.6f}'.format(1 / scores_column_[i])
            fout.write(mesh_name + ' ' + distance + '\n')
