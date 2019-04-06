#!/bin/bash
set -eux

work_dir=${1:-$(pwd)/google_speech_commands}

# download file
mkdir -p ${work_dir}
pushd ${work_dir}
wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
tar xzf speech_commands_v0.01.tar.gz
popd

# split data
output_dir=${work_dir}/splitted_data
python google_speech_commmands_dataset_to_our_format_with_split.py \
    --input_dir `realpath ${work_dir}` \
    --train_list_fullpath train.txt \
    --valid_list_fullpath valid.txt \
    --test_list_fullpath test.txt \
    --wanted_words yes,no,up,down,left,right,on,off,stop,go \
    --output_dir `realpath ${output_dir}`
echo "Dataset is prepared at ${output_dir}"
