import torch
from model.view_gcn import view_GCN, SVCNN
from torch.nn import AdaptiveAvgPool2d
import torchvision.models as models


alex = models.alexnet()

rank = 0

sv_classifier = SVCNN('test', nclasses=40, pretraining=True, cnn_name='alexnet')
# print('sv_classifier:', sv_classifier)

viewgcn = view_GCN('debug-viewgcn-alexnet', sv_classifier, nclasses=40, cnn_name='alexnet', num_views=20).to(rank)

in_data = torch.randn(60, 3, 224, 224, device=rank)

# out_base = sv_classifier(in_data)
# print('out_base.shape:', out_base.shape)

# out1 = viewgcn.net_1(in_data)
# print('out1.shape:', out1.shape)

# avg_pool2d = AdaptiveAvgPool2d(output_size=(6,6))
# out2 = avg_pool2d(out1)
# print('out2.shape:', out2.shape)

out_data, F_score, F_score2 = viewgcn(in_data)
print('out_data.shape:', out_data.shape)
print('F_score.shape:', F_score.shape)
print('F_score2.shape:', F_score2.shape)
