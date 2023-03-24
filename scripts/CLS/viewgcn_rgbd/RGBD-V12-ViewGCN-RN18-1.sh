# 2023.3.24

#   test_inst_acc: 
#   test_class_acc: 

PROJ_NAME=viewgcn_rgbd
MAIN_PROGRAM=train_rgbd.py
MODEL_NAME=view_gcn.py
BASE_MODEL_NAME=resnet18

TASK=CLS
BASE_MODEL_EPOCHS=30
EPOCHS=15
LR=0.001

DATASET=RGBD
TRAIN_PATH=data/rgbd-dataset_eval
TEST_PATH=data/rgbd-dataset_eval
NUM_CLASSES=51
NUM_VIEWS=12
BATCH_SIZE=750
BASE_MODEL_BATCH_SIZE=4500
PRINT_FREQ=10
NUM_WORKERS=3

# 1st trial
EXP_NAME=RGBD-V12-ViewGCN-RN18-1_1
pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
    --proj_name ${PROJ_NAME} \
    --exp_name ${EXP_NAME} \
    --main_program ${MAIN_PROGRAM} \
    --model_name ${MODEL_NAME} \
    --shell_name scripts/${TASK}/${PROJ_NAME}/RGBD-V12-ViewGCN-RN18-1.sh \
    --stage_one --stage_two \
    --dataset ${DATASET} \
    --num_obj_classes ${NUM_CLASSES} \
    --task ${TASK} \
    --train_path ${TRAIN_PATH} \
    --test_path ${TEST_PATH} \
    --trial_id 1 \
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
    

# trial 2-10, use pretrained alexnet weights in 1st trial 
for i in 2 3 4 5 6 7 8 9 10
do
    EXP_NAME=RGBD-V12-ViewGCN-RN18-1_$i
    pueue add -g ${PROJ_NAME} python ${MAIN_PROGRAM} \
        --proj_name ${PROJ_NAME} \
        --exp_name ${EXP_NAME} \
        --main_program ${MAIN_PROGRAM} \
        --model_name ${MODEL_NAME} \
        --shell_name scripts/${TASK}/${PROJ_NAME}/RGBD-V12-ViewGCN-RN18-1.sh \
        --stage_two \
        --dataset ${DATASET} \
        --num_obj_classes ${NUM_CLASSES} \
        --task ${TASK} \
        --train_path ${TRAIN_PATH} \
        --test_path ${TEST_PATH} \
        --trial_id $i \
        --base_model_name ${BASE_MODEL_NAME} \
        --resume --base_model_weights runs/${TASK}/${PROJ_NAME}/RGBD-V12-ViewGCN-RN18-1_1/weights/sv_model_best.pth \
        --base_model_epochs ${BASE_MODEL_EPOCHS} --epochs ${EPOCHS} \
        --lr ${LR} \
        --num_views ${NUM_VIEWS} \
        --batch_size ${BATCH_SIZE} \
        --base_model_batch_size ${BASE_MODEL_BATCH_SIZE} \
        --test_batch_size ${BATCH_SIZE} \
        --base_model_test_batch_size ${BASE_MODEL_BATCH_SIZE} \
        --print_freq ${PRINT_FREQ} \
        --num_workers ${NUM_WORKERS} \
done