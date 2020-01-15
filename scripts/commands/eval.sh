#!/usr/bin/env bash
trap 'pkill -P $$' SIGINT SIGTERM EXIT
python evaluate_audio.py --dataset_path google_speech_commands/splitted_data --dataset_split_name valid --output_name output/softmax --num_classes 12 \
--checkpoint_path work/v1/DpNet1Model/DPNET-1.0_0.0001_adam_l2 \
--num_silent 258 --augmentation_method anchored_slice_or_pad \
--background_frequency 0.0 --background_max_volume 0.0 --max_step_from_restore 30000 --batch_size 16 --no-shuffle \
--valid_type loop DpNet1Model --weight_decay 0.00001 &
wait
python evaluate_audio.py --dataset_path google_speech_commands/splitted_data --dataset_split_name test --output_name output/softmax --num_classes 12 \
--checkpoint_path work/v1/DpNet1Model/DPNET-1.0_0.0001_adam_l2/valid/accuracy/valid \
--num_silent 257 --augmentation_method anchored_slice_or_pad \
--background_frequency 0.0 --background_max_volume 0.0 --max_step_from_restore 30000 --batch_size 16 --no-shuffle \
--valid_type once DpNet1Model --weight_decay 0