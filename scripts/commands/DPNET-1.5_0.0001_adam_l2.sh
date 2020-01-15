#!/usr/bin/env bash
trap 'pkill -P $$' SIGINT SIGTERM EXIT
python train_audio.py --dataset_path google_speech_commands/splitted_data --dataset_split_name train \
--output_name output/softmax --num_classes 12 --train_dir work/v1/DpNet1Model/DPNET-1.5_0_adam_l2 \
--num_silent 1854 --augmentation_method anchored_slice_or_pad_with_shift \
--boundaries 10000 20000 --max_step_from_restore 30000 --lr_list 0.01 0.001 0.0001 \
--absolute_schedule --no-boundaries_epoch --max_to_keep 20 --step_save_checkpoint 500 --step_evaluation 500 \
--optimizer adam DpNet1Model --weight_decay 0 --width_multiplier 1.5 &
sleep 5
python evaluate_audio.py --dataset_path google_speech_commands/splitted_data --dataset_split_name valid --output_name output/softmax --num_classes 12 \
--checkpoint_path work/v1/DpNet1Model/DPNET-1.5_0_adam_l2 \
--num_silent 258 --augmentation_method anchored_slice_or_pad \
--background_frequency 0.0 --background_max_volume 0.0 --max_step_from_restore 30000 --batch_size 16 --no-shuffle \
--valid_type loop DpNet1Model --weight_decay 0 --width_multiplier 1.5 &
wait
python evaluate_audio.py --dataset_path google_speech_commands/splitted_data --dataset_split_name test --output_name output/softmax --num_classes 12 \
--checkpoint_path work/v1/DpNet1Model/DPNET-1.5_0_adam_l2/valid/accuracy/valid \
--num_silent 257 --augmentation_method anchored_slice_or_pad \
--background_frequency 0.0 --background_max_volume 0.0 --max_step_from_restore 30000 --batch_size 16 --no-shuffle \
--valid_type once DpNet1Model --weight_decay 0 --width_multiplier 1.5
