# train viewgcn classification model on modelnet40

PROJ_NAME=viewgcn_mn40
EXP_NAME=MN40-V20-ViewGCN-RN18-1
MAIN_PROGRAM=train_modelnet.py
MODEL_NAME=view_gcn.py
BASE_MODEL_NAME=resnet18

TASK=CLS

BASE_MODEL_EPOCHS=30
EPOCHS=15
LR=0.001

DATASET=ModelNet40
NUM_CLASSES=40
NUM_VIEWS=20
BATCH_SIZE=450  # 不能是60的整数倍，否则最后一个batch大小为1，前向传播报错
BASE_MODEL_BATCH_SIZE=4500
PRINT_FREQ=9
NUM_WORKERS=3


pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
    --proj_name ${PROJ_NAME} \
    --exp_name ${EXP_NAME} \
    --main_program ${MAIN_PROGRAM} \
    --model_name ${MODEL_NAME} \
    --shell_name scripts/${TASK}/${PROJ_NAME}/${EXP_NAME}.sh \
    --stage_one --stage_two \
    --dataset ${DATASET} \
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
    