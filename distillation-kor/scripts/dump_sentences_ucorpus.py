# -*- coding: utf-8 -*-

import os
import re
import sys
import argparse

from tqdm import tqdm


def parse_arguments(argv):

    parser = argparse.ArgumentParser(description='Dump UCorpus from Exbrain.')
    parser.add_argument('--root', type=str, default='./.data/exbrain-ucorpus/')
    parser.add_argument('--min_string_length', default=10, type=int)
    parser.add_argument('--dump_file', type=str)

    return parser.parse_args(argv)


def main(argv=sys.argv[1:]):

    args = parse_arguments(argv)

    filename = os.path.join(args.root, 'training_corpus_exbrain.txt')
    assert os.path.exists(filename), f'Check file existence: {filename}.'

    # Read raw data
    with open(filename, 'rb') as f:
        raw_corpus = f.readlines()

    # Keep only valid text
    lines = []
    for i, line in enumerate(tqdm(raw_corpus, desc='Extracting raw lines only')):
        if i % 3 == 0:
            lines.append(line)
    print(f"{len(lines)} lines available.")

    # Some cleaning
    processed_lines = []
    skipped = 0
    for l in tqdm(lines, desc='Cleaning'):
        try:
            l_ = l.decode('cp949')
            l_ = re.sub(r'[\n\r\t]', '', l_)  # remove str controllers
            l_ = re.sub(r'\s+', ' ', l_)      # remove duplicate whitespaces
            l_ = l_.strip()
            processed_lines.append(l_)
        except UnicodeDecodeError:
            skipped += 1
    print(f"Skipped {skipped} lines due to `UnicodeDecodeError`.")

    # Dump file to 'dump_ucorpus.txt'
    with open(args.dump_file, mode='w') as f:
        for line in tqdm(processed_lines, desc='Writing'):
            f.write(f"{line}\n")


if __name__ == '__main__':
    main()
