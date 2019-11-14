# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing script before distillation.
"""

import time
import json
import argparse
import pickle
import random
import logging

import numpy as np
from transformers import BertTokenizer, RobertaTokenizer, GPT2Tokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    parser = argparse.ArgumentParser(description="Preprocess the data to avoid re-doing it several times by (tokenization + token_to_ids).")
    parser.add_argument('--file_path', type=str, help='The path to the data. Must be in .txt format.')
    parser.add_argument('--tokenizer_type', type=str, default='bert', choices=['bert', 'roberta', 'gpt2', 'kobert'])
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-multilingual-cased', help="The tokenizer to use.")
    parser.add_argument('--dump_file', type=str, help='The dump file prefix.')
    parser.add_argument('--kobert_tokenizer_configs', type=str, default='.kobert/kobert_tokenizer_configs.json')

    args = parser.parse_args()


    logger.info(f'Loading Tokenizer ({args.tokenizer_name})')
    if args.tokenizer_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['cls_token'] # `[CLS]`
        sep = tokenizer.special_tokens_map['sep_token'] # `[SEP]`
    elif args.tokenizer_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['cls_token'] # `<s>`
        sep = tokenizer.special_tokens_map['sep_token'] # `</s>`
    elif args.tokenizer_type == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
        bos = tokenizer.special_tokens_map['bos_token'] # `<|endoftext|>`
        sep = tokenizer.special_tokens_map['eos_token'] # `<|endoftext|>`
    elif args.tokenizer_type == 'kobert':
        tokenizer_configs = json.load(open(args.kobert_tokenizer_configs, 'r'))
        # Method 1.
        # tokenizer = BertTokenizer.from_pretrained(tokenizer_configs['vocab_file'])
        # Method 2.
        tokenizer = BertTokenizer(**tokenizer_configs)
        tokenizer.max_len = 512
        bos = tokenizer.special_tokens_map['cls_token'] # `[CLS]`
        sep = tokenizer.special_tokens_map['sep_token'] # `[SEP]`

    logger.info(f'Loading text from {args.file_path}')
    with open(args.file_path, 'r', encoding='utf8') as fp:
        data = fp.readlines()


    logger.info(f'Start encoding')
    logger.info(f'{len(data)} examples to process.')

    rslt = []
    iter_ = 0
    skipped = 0
    interval = 10000
    start = time.time()
    for text in data:
        text = f'{bos} {text.strip()} {sep}'
        token_ids = tokenizer.encode(text)
        if len(token_ids) > tokenizer.max_len:
            logger.info(f'Skip example of length {len(token_ids)} > {tokenizer.max_len}')
            skipped += 1
            continue
        rslt.append(token_ids)

        iter_ += 1
        if iter_ % interval == 0:
            end = time.time()
            logger.info(f'{iter_} examples processed. - {(end-start)/interval:.2f}s/expl')
            start = time.time()

    logger.info('Finished binarization')
    logger.info(f'{len(data)} examples processed.')
    logger.info(f'{skipped} examples skipped.')

    dp_file = f'{args.dump_file}.{args.tokenizer_name}.pickle'
    rslt_ = [np.uint16(d) for d in rslt]
    random.shuffle(rslt_)
    logger.info(f'Dump to {dp_file}')
    with open(dp_file, 'wb') as handle:
        pickle.dump(rslt_, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
