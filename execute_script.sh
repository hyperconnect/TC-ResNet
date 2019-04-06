#!/bin/sh
set -eux

# setup your train and data paths
export ROOT_TRAIN_DIR=
export DATA_SPEECH_COMMANDS_V1_DIR=/data/google_audio/data_speech_commands_v0.01_newsplit
export DATA_SPEECH_COMMANDS_V2_DIR=/data/google_audio/data_speech_commands_v0.02_newsplit

## DATASET_SPLIT_NAME
# train
# valid
# test
dataset_split_name=

## AVAILABLE MODELS
# KWSModel
# Res8Model
# Res8NarrowModel
# Res15Model
# Res15NarrowModel
# DSCNNSModel
# DSCNNMModel
# DSCNNLModel
# TCResNet_8Model
# TCResNet_14Model
# TCResNet_2D8Model
model=

## AVAILABLE AUDIO PREPROCESS METHODS
# log_mel_spectrogram
# mfcc*
audio_preprocess_setting=

## AVAILABLE WINDOW SETTINGS
# 3010
# 4020
window_setting=

## WEIGHT DECAY
# 0.0
# 0.001
weight_decay=

## AVAILABLE OPTIMIZER SETTINGS
# adam
# mom
opt_setting=

## AVAILABLE LEARNING RATE SETTINGS
# s1
# l1
# l2
# l3
lr_setting=

## DEPTH MULTIPLIER
width_multiplier=

## AVAILABLE DATASETS
# v1
# v2
dataset_settings=


./scripts/script_google_audio.sh \
    ${dataset_split_name} \
    ${model} \
    ${audio_preprocess_setting} \
    ${window_setting} \
    ${weight_decay} \
    ${opt_setting} \
    ${lr_setting} \
    ${width_multiplier} \
    ${dataset_settings}
