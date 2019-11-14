# -*- coding: utf-8 -*-

import os
import re
import sys
import glob
import pickle
import argparse

from tqdm import tqdm


def parse_arguments(argv):

    parser = argparse.ArgumentParser(description='Dump text from manually collected Korean news data.')
    parser.add_argument('--root', type=str, default='./.data/news-kor/')
    parser.add_argument('--min_string_length', default=10, type=int)
    parser.add_argument('--dump_file', type=str)

    return parser.parse_args(argv)


def main(argv=sys.argv[1:]):

    args = parse_arguments(argv)

    filenames = glob.glob(os.path.join(args.root, '**/*.pickle'))
    processed_text = []

    for filename in tqdm(filenames, desc='Iterating pickle files'):

        # Open pickle files
        with open(filename, 'rb') as f:
            articles = pickle.load(f)
            assert isinstance(articles, list)

        # Process each article
        for article in articles:

            # Get text
            text = article['history'][-1]['description']

            text = re.sub(r'\S*@\S*\s', '', text)  # remove e-mail
            text = re.sub(r'[\n\r\t]', '', text)   # remove str controllers
            text = re.sub(r'\s+', ' ', text)       # remove duplicate wspaces
            text = text.strip()                    # remove redundant whitespaces

            processed_text.append(text)
        
    # Dump file to 'dump_news-kor.txt'
    with open(args.dump_file, mode='w') as f:
        for text in tqdm(processed_text, desc='Writing to single txt file'):
            f.write(f"{text}\n")

    # To open the file...
    # with open(args.dump_file, mode='r') as f:
    #     text = f.readlines()


if __name__ == '__main__':
    main()
