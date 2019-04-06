import random
import argparse

from pathlib import Path
from collections import defaultdict


UNKNOWN_WORD_LABEL = 'unknown'
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185
random.seed(RANDOM_SEED)


def check_path_existence(path, name):
    assert path.exists(), (f"{name} ({path}) does not exist!")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input_dir", type=lambda p: Path(p), required=True,
                        help="Directory as the result of `tar -zxvf <xxx.tar.gz>`.")
    parser.add_argument("--background_noise_dir", type=lambda p: Path(p), required=True,
                        help="Directory containing noise wav files")
    parser.add_argument("--test_list_fullpath", type=lambda p: Path(p), required=True,
                        help="Textfile which contains name of test wave files.")
    parser.add_argument("--valid_list_fullpath", type=lambda p: Path(p), required=True,
                        help="Textfile which contains name of validation wave files.")
    parser.add_argument("--output_dir", type=lambda p: Path(p), required=True,
                        help="Directory which will contain the result dataset.")
    parser.add_argument("--wanted_words", type=str, default="",
                        help="Comma seperated words to be categorized as foreground. Default '' means take all.")

    args = parser.parse_args()

    # validation check
    check_path_existence(args.input_dir, "Input directory")
    check_path_existence(args.background_noise_dir, "Background noise directory")
    check_path_existence(args.test_list_fullpath, "`test_list_fullpath`")
    check_path_existence(args.valid_list_fullpath, "`valid_list_fullpath`")
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
    return label == BACKGROUND_NOISE_DIR_NAME


def split_files(input_dir, valid_list_fullpath, test_list_fullpath, wanted_words):
    # load split list
    with test_list_fullpath.open("r") as fr:
        test_names = {row.strip(): True for row in fr.readlines()}

    with valid_list_fullpath.open("r") as fr:
        valid_names = {row.strip(): True for row in fr.readlines()}

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

    # iter input directory to get all wav files
    samples = {
        'train': defaultdict(list),
        'valid': defaultdict(list),
        'test': defaultdict(list),
    }

    for p in input_dir.rglob("*.wav"):
        label, filename = get_label_and_filename(p)
        if not is_noise_label(label):
            name = f"{label}/{filename}"

            if not is_valid_label(label, valid_labels):
                label = UNKNOWN_WORD_LABEL

            if test_names.get(name, False):
                samples["test"][label].append(p)
            elif valid_names.get(name, False):
                samples["valid"][label].append(p)
            else:
                samples["train"][label].append(p)

    has_unknown = all([UNKNOWN_WORD_LABEL in label_samples for split, label_samples in samples.items()])
    for split, label_samples in samples.items():
        if has_unknown:
            assert len(label_samples) == len(valid_labels) + 1, f"{set(label_samples)} == {valid_labels}"
        else:
            assert len(label_samples) == len(valid_labels), f"{set(label_samples)} == {valid_labels}"

    # number of samples
    num_train = sum(map(lambda kv: len(kv[1]), samples["train"].items()))
    num_valid = sum(map(lambda kv: len(kv[1]), samples["valid"].items()))
    num_test = sum(map(lambda kv: len(kv[1]), samples["test"].items()))

    num_samples = num_train + num_valid + num_test

    print(f"Num samples with train / valid / test split: {num_train} / {num_valid} / {num_test}")
    print(f"Total {num_samples} samples, {len(valid_labels)} labels")
    assert num_train > num_test
    assert num_train > num_valid

    # filtering unknown samples
    # the number of unknown samples -> mean of all other samples
    if has_unknown:
        mean_num_samples_per_label = dict()
        for split, label_samples in samples.items():
            s = 0
            c = 0
            for label, sample_list in label_samples.items():
                if label != UNKNOWN_WORD_LABEL:
                    s += len(sample_list)
                    c += 1

            m = int(s / c)

            unknown_samples = label_samples[UNKNOWN_WORD_LABEL]
            if len(unknown_samples) > m:
                random.shuffle(unknown_samples)
                label_samples[UNKNOWN_WORD_LABEL] = unknown_samples[:m]

        # number of samples
        print("After Filtered:")
        num_train = sum(map(lambda kv: len(kv[1]), samples["train"].items()))
        num_valid = sum(map(lambda kv: len(kv[1]), samples["valid"].items()))
        num_test = sum(map(lambda kv: len(kv[1]), samples["test"].items()))

        num_samples = num_train + num_valid + num_test

        print(f"Num samples with train / valid / test split: {num_train} / {num_valid} / {num_test}")
        print(f"Total {num_samples} samples, {len(valid_labels)} labels")

    return samples


def generate_dataset(split_samples, background_noise_dir, output_dir):
    # make output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # link splitted samples
    for split, label_samples in split_samples.items():
        base_dir = output_dir / split
        base_dir.mkdir()

        # make labels
        for label in label_samples:
            label_dir = base_dir / label
            label_dir.mkdir()

        # link samples
        for label, samples in label_samples.items():
            for sample_path in samples:
                # do not use label from get_label_and_filename.
                # we already replace some of them as UNKNOWN_WORD_LABEL
                old_label, filename = get_label_and_filename(sample_path)

                if label == UNKNOWN_WORD_LABEL:
                    filename = f"{old_label}_{filename}"

                target_path = base_dir / label / filename
                target_path.symlink_to(sample_path)

        # link background_noise
        if split == "train":
            noise_dir = base_dir / BACKGROUND_NOISE_DIR_NAME
            noise_dir.symlink_to(background_noise_dir, target_is_directory=True)

        print(f"Make {base_dir} done.")


if __name__ == "__main__":
    args = parse_arguments()

    samples = split_files(args.input_dir, args.valid_list_fullpath, args.test_list_fullpath, args.wanted_words)
    generate_dataset(samples, args.background_noise_dir, args.output_dir)
