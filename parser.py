import argparse


parser = argparse.ArgumentParser(description='Multi View Transformer for 3D Shape Analysis')

# --------- initialize project
parser.add_argument('--proj_name', type=str, default='ViewGCN-DDP', metavar='N',
                    help='Name of the project')
parser.add_argument('--exp_name', type=str, default='1CL1SB6SL_4', metavar='N',
                    help='Name of the experiment')
parser.add_argument('--main_program', type=str, default='main.py', metavar='N',
                    help='Name of main program')
parser.add_argument('--model_name', type=str, default='transformer.py', metavar='N',
                    help='Name of model file')
parser.add_argument('--shell_name', type=str, default='scripts/run.sh', metavar='N',
                    help='Name of shell file')

# --------- training settings
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of episode to train ')
parser.add_argument('--base_model_epochs', type=int, default=30, metavar='N',
                    help='number of episode to train base image model')
parser.add_argument('--batch_size', type=int, default=96, metavar='batch_size',
                    help='Size of batch')
parser.add_argument('--base_model_batch_size', type=int, default=2400, metavar='batch_size',
                    help='Size of batch in base model')
parser.add_argument('--test_batch_size', type=int, default=96, metavar='test_batch_size',
                    help='Size of test batch')
parser.add_argument('--base_model_test_batch_size', type=int, default=2400, metavar='test_batch_size',
                    help='Size of test batch in base model')
parser.add_argument('--num_workers', type=int, default=0, metavar='num_workers',
                    help='number of processes to load data in host memory')
parser.add_argument('--save_freq', type=int, default=50, help='save frequency')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
parser.add_argument('--task', type=str, default='CLS', choices=['CLS', 'RET'], 
                    help='Task: Classification or Retrieval')
parser.add_argument('--eval', action='store_true',  help='test the model')

parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
parser.add_argument('--base_model_name', type=str, default='resnet18', 
                    help='base model for image feature extraction')
parser.add_argument('--base_feature_dim', type=int, default=512, help='feature dimension of base model')
parser.add_argument('--base_pretrain', type=bool, default=True, help='flag to whether use pretrained image feature extractor')
parser.add_argument('--base_model_weights', type=str, default='sv_model_best.pth', metavar='N',
                    help='saved base image model name')
parser.add_argument('--stage_one', action="store_true", help='flag to run stage one')
parser.add_argument('--stage_two', action="store_true", help='flag to run stage two')
parser.add_argument('--num_views', type=int, default=20, help='number of views per 3d object, its value is <= total_num_views')
parser.add_argument('--total_num_views', type=int, default=20, help='maximum number of views per 3d object')

# --------- shape retrieval on ShapeNet Core55
parser.add_argument('--train_label', type=str, default='train.csv', help='train label file')
parser.add_argument('--test_label', type=str, default='test.csv', help='test label file')
parser.add_argument('--val_label', type=str, default='val.csv', help='val label file')
parser.add_argument('--shrec_version', type=str, default='normal', choices=['normal', 'perturbed'], 
                    help='shrec17 version')

# --------- optimization settings
parser.add_argument('--optim', type=str, default='adamw', metavar='N',
                    choices=['sgd', 'adam', 'adamw'], help='optimizer to choose')
parser.add_argument('--base_model_optim', type=str, default='sgd', metavar='N',
                    choices=['sgd', 'adam', 'adamw'], help='optimizer to choose')
parser.add_argument('--label_smoothing', type=float, default=0.1, 
                    help='label smoothing for soft classification')                 
parser.add_argument('--base_model_lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01, 0.1 if using sgd)')
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001, 0.1 if using sgd)')
parser.add_argument('--max_lr', type=float, default=0.1, metavar='LR',
                    help='maximum learning rate')
parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                    help='minimum learning rate')
parser.add_argument('--warm_epochs', type=int, default=10,
                    help='number of iterations for the first restart')
parser.add_argument('--patience', type=int, default=10,
                    help='number of epochs with no improvement after which learning rate will be reduced')
parser.add_argument('--step_size', type=int, default=30,
                    help='adapt learning rate every step_size epochs during training')
parser.add_argument('--gamma', type=float, default=0.1, help='lr = lr * gamma, every step_size epochs')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--lr_scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'coswarm', 'plateau', 'step'],
                    help='LR Scheduler to use, [cos, step, coswarm, plateau]')
parser.add_argument('--base_model_lr_scheduler', type=str, default='cos', metavar='N',
                    choices=['cos', 'coswarm', 'plateau', 'step'],
                    help='LR Scheduler to use, [cos, step, coswarm, plateau]')
parser.add_argument('--freeze_base_model', action='store_true', help='flag to freeze base model')

# --------- training on single GPU device 
parser.add_argument('--no_cuda', type=bool, default=False,
                    help='enables CUDA training')
parser.add_argument('--gpu_id', type=int, default=0, help='specify the GPU device'
                    'to train of finetune model')

# --------- distributed training 
parser.add_argument('--backend', type=str, default='nccl', help='DDP communication backend')
parser.add_argument('--world_size', type=int, default=6, help='number of GPUs')
parser.add_argument('--master_addr', type=str, default='localhost', help='ip of master node')
parser.add_argument('--master_port', type=str, default='12355', help='port of master node')
parser.add_argument('--rank', type=int, default=0, help='the rank for current GPU or process, '
                    'ususally one process per GPU')

# --------- dataset
parser.add_argument('--dataset', type=str, default='ModelNet40', help='the dataset used for '
                    'evaluating the pretrained model')
parser.add_argument('--num_obj_classes', type=int, default=40, help='number of object classes')
parser.add_argument('--train_path', type=str, default='data/modelnet40v2png_ori4', help='path of train dataset')
parser.add_argument('--test_path', type=str, default='data/modelnet40v2png_ori4', help='path of test dataset')

# --------- wandb settings
parser.add_argument('--wb_url', type=str, default="http://localhost:28282", help='wandb server url')
parser.add_argument('--wb_key', type=str, default="", help='wandb login key')


args = parser.parse_args()
