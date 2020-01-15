#!/bin/bash
# set -eux

base_dir=$(cd $(dirname $0); pwd)
work_dir=${1:-$(pwd)/google_speech_commands}
echo ${base_dir}
echo ${work_dir}


# split data
output_dir=${work_dir}/splitted_data
python ${base_dir}/google_speech_commmands_dataset_to_our_format_with_split.py \
    --input_dir `realpath ${work_dir}` \
    --train_list_fullpath ${base_dir}/train.txt \
    --valid_list_fullpath ${base_dir}/valid.txt \
    --test_list_fullpath ${base_dir}/test.txt \
    --wanted_words yes,no,up,down,left,right,on,off,stop,go \
    --output_dir `realpath ${output_dir}`
echo "Dataset is prepared at ${output_dir}"
