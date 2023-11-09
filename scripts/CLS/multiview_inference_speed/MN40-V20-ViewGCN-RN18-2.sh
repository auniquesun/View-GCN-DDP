# 2023/11/07

# NOTE 记录 train/test time，之前 Trainer.py 计算时间的方式有问题，这一版改过来了
#   0. train viewgcn classification model on modelnet40
#   1. stage_one 跑的 train_interval/test_interval 能看作 SVCNN 的结果
#         train_interval * num_views 能得到 MVCNN 的结果，test_interval * num_views 类推
#   2. stage_two 跑的 train_interval/test_interval 是 viewgcn 的结果

PROJ_NAME=multiview_inference_speed
EXP_NAME=MN40-V20-ViewGCN-RN18-2
MAIN_PROGRAM=train_modelnet.py
MODEL_NAME=view_gcn.py
BASE_MODEL_NAME=resnet18

TASK=CLS

BASE_MODEL_EPOCHS=30
EPOCHS=15
LR=0.001

DATASET=ModelNet40
TRAIN_PATH=data/${DATASET}
TEST_PATH=data/${DATASET}
NUM_CLASSES=40
NUM_VIEWS=20
BATCH_SIZE=84  # 不能是60的整数倍，否则最后一个batch大小为1，前向传播报错
BASE_MODEL_BATCH_SIZE=2400
PRINT_FREQ=40
NUM_WORKERS=2


pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
    --proj_name ${PROJ_NAME} \
    --exp_name ${EXP_NAME} \
    --main_program ${MAIN_PROGRAM} \
    --model_name ${MODEL_NAME} \
    --shell_name scripts/${TASK}/${PROJ_NAME}/${EXP_NAME}.sh \
    --stage_one --stage_two \
    --dataset ${DATASET} \
    --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} \
    --num_obj_classes ${NUM_CLASSES} \
    --task ${TASK} \
    --base_model_name ${BASE_MODEL_NAME} \
    --resume --base_model_weights runs/${TASK}/${PROJ_NAME}/${EXP_NAME}/weights/sv_model_best.pth \
    --base_model_epochs ${BASE_MODEL_EPOCHS} --epochs ${EPOCHS} \
    --lr ${LR} \
    --num_views ${NUM_VIEWS} \
    --batch_size ${BATCH_SIZE} \
    --base_model_batch_size ${BASE_MODEL_BATCH_SIZE} \
    --test_batch_size ${BATCH_SIZE} \
    --base_model_test_batch_size ${BASE_MODEL_BATCH_SIZE} \
    --print_freq ${PRINT_FREQ} \
    --num_workers ${NUM_WORKERS} \
    