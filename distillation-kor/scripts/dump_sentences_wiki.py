# -*- coding: utf-8 -*-

import os
import sys
import json
import glob
import argparse

from tqdm import tqdm


def parse_arguments(argv):

    parser = argparse.ArgumentParser(description='Dump text.')
    parser.add_argument('--root', type=str)
    parser.add_argument('--min_string_length', default=10, type=int)
    parser.add_argument('--dump_file', type=str)

    return parser.parse_args(argv)


def main(argv=sys.argv[1:]):

    args = parse_arguments(argv)

    filenames = glob.glob(os.path.join(args.root, '**/wiki_*'))
    processed_lines = []

    for filename in tqdm(filenames, desc='Processing'):

        # Open file
        with open(filename, encoding='utf-8') as f:
            articles = f.readlines()

        # string -> dict
        articles = [json.loads(a) for a in articles]

        # dict -> dict.text
        articles = [a.get('text') for a in articles]

        # Split article to sequences
        split_token = "\n\n"
        articles = [a.split(split_token) for a in articles]

        # Save each sentence
        for article in articles:
            for line in article:
                if len(line) < args.min_string_length:
                    continue
                else:
                    processed_lines.append(line)

    # Dump file to 'dump_kowiki.txt'
    with open(args.dump_file, mode='w') as f:
        for line in tqdm(processed_lines, desc='Writing'):
            f.write(f"{line}\n")

    # To open the file...
    # with open(args.dump_file, mode='r') as f:
    #    lines = f.readlines()


if __name__ == '__main__':
    main()
