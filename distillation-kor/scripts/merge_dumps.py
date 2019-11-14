# -*- coding: utf-8 -*-

import os
import sys
import argparse
import logging

import numpy as np
from tqdm import tqdm


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments(argv):

    parser = argparse.ArgumentParser(description='merge txt files with one sequence per line.')
    parser.add_argument('--dump_root', type=str, default='./.data/')
    parser.add_argument('--in_files', nargs='+', required=True)
    parser.add_argument('--out_file', required=True)

    return parser.parse_args(argv)


def main(argv=sys.argv[1:]):

    args = parse_arguments(argv)

    logger.info('Merging the following dump files:')
    logger.info(', '.join(args.in_files))

    out = []
    for in_file in args.in_files:
        with open(os.path.join(args.dump_root, in_file), 'r') as f:
            sentences = f.readlines()
            out.extend(sentences)
    logger.info('Loaded all input files.')

    sample_size = 1000
    sample_file = os.path.join(args.dump_root, args.out_file.replace('.txt', '') + '_sample.txt')
    logger.info(f'Writing a sample dump to {sample_file}.')
    with open(sample_file, mode='w') as f:
        for idx in tqdm(np.random.randint(0, len(out), sample_size), desc='Writing sample'):
            f.write(out[idx])

    merged_file = os.path.join(args.dump_root, args.out_file)
    logger.info(f'Writing the merged dump to {merged_file}.')
    with open(merged_file, mode='w') as f:
        for sent in tqdm(out, desc='Writing'):
            f.write(sent)


if __name__ == '__main__':
    main()
