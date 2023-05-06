# 2023.5.5

# NOTE 相比上次实验，这次 train/test time 统计的操作发生了变化
#   train 包含前向、反向传播求梯度、优化器根据梯度更新参数时间
#   test 仅包含前向传播得到预测结果时间

# train viewgcn classification model on modelnet40

WB_KEY=local-blabla     # replace with your wandb key
PROJ_NAME=multiview_time_comparison
EXP_NAME=MN40-V20-ViewGCN-RN18-3
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
BATCH_SIZE=96
BASE_MODEL_BATCH_SIZE=2400
PRINT_FREQ=50
NUM_WORKERS=2


pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
    --wb_key ${WB_KEY} \
    --proj_name ${PROJ_NAME} \
    --exp_name ${EXP_NAME} \
    --main_program ${MAIN_PROGRAM} \
    --model_name ${MODEL_NAME} \
    --shell_name scripts/${TASK}/${PROJ_NAME}/${EXP_NAME}.sh \
    --stage_two \
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
    