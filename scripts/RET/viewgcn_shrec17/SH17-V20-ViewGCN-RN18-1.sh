# train viewgcn retrieval model on shrec17

WB_KEY=local-blabla     # replace with your wandb key
PROJ_NAME=viewgcn_shrec17
EXP_NAME=SH17-V20-ViewGCN-RN18-1
MAIN_PROGRAM=train_shrec17.py
MODEL_NAME=view_gcn.py
BASE_MODEL_NAME=resnet18

TASK=RET
TRAIN_LABEL=train.csv
TEST_LABEL=test.csv
VERSION=normal

BASE_MODEL_EPOCHS=30
EPOCHS=15

DATASET=SHREC17
TRAIN_PATH=data/${DATASET}
TEST_PATH=data/${DATASET}

NUM_CLASSES=55
NUM_VIEWS=20
WORLD_SIZE=6
BATCH_SIZE=72
BASE_MODEL_BATCH_SIZE=1632  # 是试出来的
PRINT_FREQ=100
NUM_WORKERS=2

pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
    --wb_key ${WB_KEY} \
    --proj_name ${PROJ_NAME} \
    --exp_name ${EXP_NAME} \
    --main_program ${MAIN_PROGRAM} \
    --model_name ${MODEL_NAME} \
    --shell_name scripts/${TASK}/${PROJ_NAME}/${EXP_NAME}.sh \
    --stage_one --stage_two \
    --dataset ${DATASET} \
    --train_path ${TRAIN_PATH} --test_path ${TEST_PATH} \
    --num_obj_classes ${NUM_CLASSES} \
    --task ${TASK} --train_label ${TRAIN_LABEL} --test_label ${TEST_LABEL} --shrec_version ${VERSION} \
    --base_model_name ${BASE_MODEL_NAME} \
    --resume --base_model_weights runs/RET/${PROJ_NAME}/${EXP_NAME}/weights/sv_model_best.pth \
    --base_model_epochs ${BASE_MODEL_EPOCHS} --epochs ${EPOCHS} \
    --world_size ${WORLD_SIZE} \
    --num_views ${NUM_VIEWS} \
    --batch_size ${BATCH_SIZE} \
    --base_model_batch_size ${BASE_MODEL_BATCH_SIZE} \
    --test_batch_size ${BATCH_SIZE} \
    --base_model_test_batch_size ${BASE_MODEL_BATCH_SIZE} \
    --print_freq ${PRINT_FREQ} \
    --num_workers ${NUM_WORKERS} \
