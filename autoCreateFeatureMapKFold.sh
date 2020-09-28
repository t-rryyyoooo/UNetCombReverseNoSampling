#!/bin/bash

# Input 
readonly INPUT_DIRECTORY="input"
echo -n "Is json file name createFeatureMapKFold.json?[y/n]:"
read which
while [ ! $which = "y" -a ! $which = "n" ]
do
 echo -n "Is json file name the same as this file name?[y/n]:"
 read which
done

# Specify json file.
if [ $which = "y" ];then
 JSON_NAME="createFeatureMapKFold.json"
else
 echo -n "JSON_FILE_NAME="
 read JSON_NAME
fi

# From json file, read required variables.
readonly JSON_FILE="${INPUT_DIRECTORY}/${JSON_NAME}"
readonly DATA_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".data_directory"))
readonly MODELWEIGHT_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".modelweight_directory"))
readonly SAVE_DIRECTORY=$(eval echo $(cat ${JSON_FILE} | jq -r ".save_directory"))
readonly IMAGE_NAME=$(cat ${JSON_FILE} | jq -r ".image_name")
readonly MASK_NAME=$(cat ${JSON_FILE} | jq -r ".mask_name")
readonly MODEL_NAME=$(cat ${JSON_FILE} | jq -r ".model_name")

readonly IMAGE_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".image_patch_size")
readonly LABEL_PATCH_SIZE=$(cat ${JSON_FILE} | jq -r ".label_patch_size")
readonly IMAGE_PATCH_WIDTH=$(cat ${JSON_FILE} | jq -r ".image_patch_width")
readonly LABEL_PATCH_WIDTH=$(cat ${JSON_FILE} | jq -r ".label_patch_width")
readonly IS_LABEL=$(cat ${JSON_FILE} | jq -r ".is_label")
readonly KFOLD_LIST=$(cat ${JSON_FILE} | jq -r ".kfold_list[]")
readonly NUM_ARRAY=$(cat ${JSON_FILE} | jq -r ".num_array[]")
readonly LOG_FILE=$(eval echo $(cat ${JSON_FILE} | jq -r ".log_file"))

# Make directory to save LOG.
echo ${LOG_FILE}
mkdir -p `dirname ${LOG_FILE}`
date >> $LOG_FILE

for kfold in ${KFOLD_LIST[@]}
do
    SAVE_PATH="${SAVE_DIRECTORY}/${kfold}"
    modelweight_path="${MODELWEIGHT_DIRECTORY}/${kfold}/${MODEL_NAME}"
    for number in ${NUM_ARRAY[@]}
    do

        if [ $MASK_NAME = "No" ]; then
            mask=""
        else
            mask_path="${DATA_DIRECTORY}/case_${number}/${MASK_NAME}"
            mask="--mask_path ${mask_path}"
        fi

        if [ $IS_LABEL = "No" ]; then
            is_label=""
        else
            is_label="--is_label"
        fi


     image_path="${DATA_DIRECTORY}/case_${number}/${IMAGE_NAME}"
     save_path="${SAVE_PATH}/case_${number}"

     echo "image_path:${image_path}"
     echo "MODELWEIGHT_PATH:${modelweight_path}"
     echo "save_path:${save_path}"
     echo "IMAGE_PATCH_SIZE:${IMAGE_PATCH_SIZE}"
     echo "LABEL_PATCH_SIZE:${LABEL_PATCH_SIZE}"
     echo "IMAGE_PATCH_WIDTH:${IMAGE_PATCH_WIDTH}"
     echo "LABEL_PATCH_WIDTH:${LABEL_PATCH_WIDTH}"
     echo "MASK_NAME:${MASK_NAME}"
     echo "is_label:${IS_LABEL}"

     python3 createFeatureMap.py ${image_path} ${modelweight_path} ${save_path} --image_patch_size ${IMAGE_PATCH_SIZE} --label_patch_size ${LABEL_PATCH_SIZE} --image_patch_width ${IMAGE_PATCH_WIDTH} --label_patch_width ${LABEL_PATCH_WIDTH} ${mask} ${is_label}
     # Judge if it works.
     if [ $? -eq 0 ]; then
      echo "case_${number} done."
     
     else
      echo "case_${number}" >> $LOG_FILE
      echo "case_${number} failed"
     
     fi

    done
done


