#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name pipliner.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file path.
if [ $which = "y" ];then
 JSON_NAME="pipliner.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"

# Training input
readonly IMAGE_PATH_1=$(cat ${JSON_FILE} | jq -r ".image_path_1")
readonly IMAGE_PATH_2=$(cat ${JSON_FILE} | jq -r ".image_path_2")
readonly LABEL_PATH=$(cat ${JSON_FILE} | jq -r ".label_path")
readonly MODEL_SAVEPATH=$(cat ${JSON_FILE} | jq -r ".model_savepath")
readonly TRAIN_LISTS=$(cat ${JSON_FILE} | jq -r ".train_lists")
readonly VAL_LISTS=$(cat ${JSON_FILE} | jq -r ".val_lists")
readonly LOG=$(cat ${JSON_FILE} | jq -r ".LOG")
readonly IN_CHANNEL_MAIN=$(cat ${JSON_FILE} | jq -r ".in_channel_main")
readonly IN_CHANNEL_FINAL=$(cat ${JSON_FILE} | jq -r ".in_channel_final")
readonly NUM_CLASS=$(cat ${JSON_FILE} | jq -r ".num_class")
readonly LR=$(cat ${JSON_FILE} | jq -r ".lr")
readonly BATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".batch_size")
readonly num_workers=$(cat ${JSON_FILE} | jq -r ".num_workers")
readonly EPOCH=$(cat ${JSON_FILE} | jq -r ".epoch")
readonly GPU_IDS=$(cat ${JSON_FILE} | jq -r ".gpu_ids")
readonly api_keys=$(cat ${JSON_FILE} | jq -r ".api_keys")
readonly PROJECT_NAME=$(cat ${JSON_FILE} | jq -r ".project_name")
readonly EXPERIMENT_NAME=$(cat ${JSON_FILE} | jq -r ".experiment_name")

readonly KEYS=$(cat ${JSON_FILE} | jq -r ".train_lists | keys[]")

# Segmentation input
readonly TEST_LISTS=$(cat ${JSON_FILE} | jq -r ".test_lists")

 TEST_LIST=$(echo $TEST_LISTS | jq -r ".$key")
 test_list=(${TEST_LIST// / })
all_patients=""

for key in ${KEYS[@]}
do 
 echo $key
 train_list=$(echo $TRAIN_LISTS | jq -r ".$key")
 val_list=$(echo $VAL_LISTS | jq -r ".$key")
 image_path_2="${IMAGE_PATH_2}/${key}"
 model_savepath="${MODEL_SAVEPATH}/${key}"
 log="${LOG}/${key}"
 experiment_name="${EXPERIMENT_NAME}_${key}"

 echo "---------- Training ----------"
 echo "image_path_1:${IMAGE_PATH_1}"
 echo "image_path_2:${image_path_2}"
 echo "label_path:${LABEL_PATH}"
 echo "model_savepath:${model_savepath}"
 echo "train_list:${train_list}"
 echo "val_list:${val_list}"
 echo "log:${log}"
 echo "IN_CHANNEL_MAIN:${IN_CHANNEL_MAIN}"
 echo "IN_CHANNEL_FINAL:${IN_CHANNEL_FINAL}"
 echo "NUM_CLASS:${NUM_CLASS}"
 echo "LR:${LR}"
 echo "BATCH_SIZE:${BATCH_SIZE}"
 echo "NUM_WORKERS:${NUM_WORKERS}"
 echo "EPOCH:${EPOCH}"
 echo "GPU_IDS:${GPU_IDS}"
 echo "API_KEYS:${API_KEYS}"
 echo "PROJECT_NAME:${PROJECT_NAME}"
 echo "EXPERIMENT_NAME:${EXPERIMENT_NAME}"

 python3 train.py ${IMAGE_PATH_1} ${image_path_2} ${LABEL_PATH} ${model_savepath} --train_list ${train_list} --val_list ${val_list} --log ${log} --in_channel_main ${IN_CHANNEL_MAIN} --in_channel_final ${IN_CHANNEL_FINAL} --num_class ${NUM_CLASS} --lr ${LR} --batch_size ${BATCH_SIZE} --num_workers ${NUM_WORKERS} --epoch ${EPOCH} --gpu_ids ${GPU_IDS} --api_keys ${API_KEYS} --project_name ${PROJECT_NAME} --experiment_name ${EXPERIMENT_NAME}


 if [ $? -ne 0 ];then
  exit 1
 fi

 model="${model_savepath}/${MODEL_NAME}"
 echo "---------- Segmentation ----------"
 echo ${test_list[@]}
 for number in ${test_list[@]}
 do
  image="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
  save="${save_directory}/case_${number}/${SAVE_NAME}"

  echo "Image:${image}"
  echo "Model:${model}"
  echo "Save:${save}"
  echo "PATCH_SIZE:${PATCH_SIZE}"
  echo "PLANE_SIZE:${PLANE_SIZE}"
  echo "OVERLAP:${OVERLAP}"
  echo "NUM_REP:${NUM_REP}"
  echo "GPU_IDS:${GPU_IDS}"

#python3 segmentation.py $image $model $save --patch_size ${PATCH_SIZE} --plane_size ${PLANE_SIZE} --overlap ${OVERLAP} --num_rep ${NUM_REP} -g ${GPU_IDS}

  if [ $? -ne 0 ];then
   exit 1
  fi

 done

 all_patients="${all_patients}${TEST_LIST} "
done

 echo "---------- Caluculation ----------"
 echo "TRUE_DIRECTORY:${DATA_DIRECTORY}"
 echo "PREDICT_DIRECTORY:${save_directory}"
 echo "CSV_SAVEPATH:${CSV_SAVEPATH}"
 echo "All_patients:${all_patients[@]}"
 echo "NUM_CLASS:${NUM_CLASS}"
 echo "CLASS_LABEL:${CLASS_LABEL}"
 echo "TRUE_NAME:${TRUE_NAME}"
 echo "PREDICT_NAME:${PREDICT_NAME}"


python3 caluculateDICE.py ${DATA_DIRECTORY} ${save_directory} ${CSV_SAVEPATH} ${all_patients} --classes ${NUM_CLASS} --class_label ${CLASS_LABEL} --true_name ${TRUE_NAME} --predict_name ${PREDICT_NAME} 

 if [ $? -ne 0 ];then
  exit 1
 fi

 echo "---------- Logging ----------"
#python3 logger.py ${JSON_FILE}
 echo Done.


