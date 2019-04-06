"""
python scripts/google_speech_commmands_dataset_to_our_format_with_split.py \
    --input_dir "/data/data/data_speech_commands_v0.01" \
    --train_list_fullpath "/data/data/data_speech_commands_v0.01/newsplit/training_list.txt" \
    --valid_list_fullpath "/data/data/data_speech_commands_v0.01/newsplit/validation_list.txt" \
    --test_list_fullpath "/data/data/data_speech_commands_v0.01/newsplit/testing_list.txt" \
    --wanted_words "yes,no,up,down,left,right,on,off,stop,go" \
    --output_dir "/data/data/google_audio/data_speech_commands_v0.01"
"""
import random
import argparse

from pathlib import Path
from collections import defaultdict


UNKNOWN_WORD_LABEL = 'unknown'
BACKGROUND_NOISE_LABEL = '_silence_'
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'


def check_path_existence(path, name):
    assert path.exists(), (f"{name} ({path}) does not exist!")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input_dir", type=lambda p: Path(p), required=True,
                        help="Directory as the result of `tar -zxvf <xxx.tar.gz>`.")
    parser.add_argument("--test_list_fullpath", type=lambda p: Path(p), required=True,
                        help="Textfile which contains name of test wave files.")
    parser.add_argument("--valid_list_fullpath", type=lambda p: Path(p), required=True,
                        help="Textfile which contains name of validation wave files.")
    parser.add_argument("--train_list_fullpath", type=lambda p: Path(p), required=True,
                        help="Textfile which contains name of train wave files.")
    parser.add_argument("--output_dir", type=lambda p: Path(p), required=True,
                        help="Directory which will contain the result dataset.")
    parser.add_argument("--wanted_words", type=str, default="",
                        help="Comma seperated words to be categorized as foreground. Default '' means take all.")

    args = parser.parse_args()

    # validation check
    check_path_existence(args.input_dir, "Input directory")
    check_path_existence(args.test_list_fullpath, "`test_list_fullpath`")
    check_path_existence(args.valid_list_fullpath, "`valid_list_fullpath`")
    check_path_existence(args.train_list_fullpath, "`valid_list_fullpath`")
    assert not args.output_dir.exists() or len([p for p in args.output_dir.iterdir()]) == 0, (
        f"Output directory ({args.output_dir}) should be empty!")

    return args


def get_label_and_filename(p):
    parts = p.parts
    label, filename = parts[-2], parts[-1]
    return label, filename


def is_valid_label(label, valid_labels):
    if valid_labels:
        is_valid = label in valid_labels
    else:
        is_valid = True

    return is_valid


def is_noise_label(label):
    return label == BACKGROUND_NOISE_DIR_NAME or label == BACKGROUND_NOISE_LABEL


def process_files(input_dir, train_list_fullpath, valid_list_fullpath, test_list_fullpath, wanted_words,
                  output_dir):
    # load split list
    data = {
        "train": [],
        "valid": [],
        "test": [],
    }

    list_fullpath = {
        "train": train_list_fullpath,
        "valid": valid_list_fullpath,
        "test": test_list_fullpath,
    }

    for split in data:
        with list_fullpath[split].open("r") as fr:
            for row in fr.readlines():
                label, filename = get_label_and_filename(Path(row.strip()))
                data[split].append((label, filename))

    # set labels
    if len(wanted_words) > 0:
        valid_labels = set(wanted_words.split(","))
    else:
        valid_labels = None

    labels = list()
    for p in input_dir.iterdir():
        if p.is_dir() and is_valid_label(p.name, valid_labels) and not is_noise_label(p.name):
            labels.append(p.name)
    assert len(set(labels)) == len(labels), f"{len(set(labels))} == {len(labels)}"

    # update valid_labels
    if len(wanted_words) > 0:
        len(valid_labels) == len(labels)
    valid_labels = set(labels)
    print(f"Valid Labels: {valid_labels}")

    # make dataset!
    # make output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for split, lst in data.items():
        # for each split
        base_dir = output_dir / split
        base_dir.mkdir()

        # make labels for valid + unknown
        for label in list(valid_labels) + [UNKNOWN_WORD_LABEL]:
            label_dir = base_dir / label
            label_dir.mkdir()

        # link files
        noise_count = 0
        for label, filename in lst:
            source_path = input_dir / label / filename

            if is_noise_label(label):
                noise_count += 1
            else:
                if not is_valid_label(label, valid_labels):
                    filename = f"{label}_{filename}"
                    label = UNKNOWN_WORD_LABEL

                target_path = base_dir / label / filename
                target_path.symlink_to(source_path)

        # report number of noise
        print(f"[{split}] Num of silences: {noise_count}")

        # link noise
        source_noise_dir = input_dir / BACKGROUND_NOISE_DIR_NAME
        target_noise_dir = base_dir / BACKGROUND_NOISE_DIR_NAME
        target_noise_dir.symlink_to(source_noise_dir, target_is_directory=True)


if __name__ == "__main__":
    args = parse_arguments()

    process_files(args.input_dir,
                  args.train_list_fullpath,
                  args.valid_list_fullpath,
                  args.test_list_fullpath,
                  args.wanted_words,
                  args.output_dir)
