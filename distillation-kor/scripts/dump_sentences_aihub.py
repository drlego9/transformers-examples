# -*- coding: utf-8 -*-

import os
import sys
import argparse

import pandas as pd

from tqdm import tqdm


def parse_arguments(argv):

    # Data can be found at: http://www.aihub.or.kr/content/605
    parser = argparse.ArgumentParser(description='Dump Korean text from AIHub Translation data.')
    parser.add_argument('--root', type=str, default='./.data/aihub-koen/')
    parser.add_argument('--min_string_length', default=10, type=int)
    parser.add_argument('--dump_file', type=str)

    return parser.parse_args(argv)


def main(argv=sys.argv[1:]):

    args = parse_arguments(argv)

    f2col = {
        '3.문어체-뉴스.xlsx': '한국어',
        '4.문어체-한국문화.xlsx': '원문',
        '5.문어체-조례.xlsx': '원문',
    }

    result = []
    for filename, col in f2col.items():

        # Open excel file with pandas and retrieve the column values
        sentences = pd.read_excel(os.path.join(args.root, filename))[col]
        sentences = sentences.tolist()

        # Save them (check length)
        for sentence in sentences:
            if len(sentence) < args.min_string_length:
                continue
            else:
                result.append(sentence)

    # Dump file to 'dump_aihub.txt'
    with open(args.dump_file, mode='w') as f:
        for sent in tqdm(sentences, desc='Writing'):
            f.write(f"{sent}\n")

    # To open the file...
    # with open(args.dump_file, mode='r') as f:
    #     sentences = f.readlines()


if __name__ == '__main__':
    main()
