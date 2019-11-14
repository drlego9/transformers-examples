# -*- coding: utf-8 -*-

import os
import sys
import json
import hashlib
import argparse
import requests


kobert_models = {
    'pytorch_kobert': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/pytorch/pytorch_kobert_2439f391a6.params',
        'fname': 'pytorch_kobert_2439f391a6.params',
        'chksum': '2439f391a6'
    },
    'vocab': {
        'url':
        'https://kobert.blob.core.windows.net/models/kobert/vocab/kobertvocab_f38b8a4d6d.json',
        'fname': 'kobertvocab_f38b8a4d6d.json',
        'chksum': 'f38b8a4d6d'
    }
}


def download(url, filename, chksum, cachedir='.kobert/'):
    f_cachedir = os.path.expanduser(cachedir)
    os.makedirs(f_cachedir, exist_ok=True)
    file_path = os.path.join(f_cachedir, filename)
    if os.path.isfile(file_path):
        if hashlib.md5(open(file_path,
                            'rb').read()).hexdigest()[:10] == chksum:
            print('using cached model')
            return file_path
    with open(file_path, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write('\r[{}{}]'.format('█' * done,
                                                   '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')
    assert chksum == hashlib.md5(open(
        file_path, 'rb').read()).hexdigest()[:10], 'corrupted file!'
    return file_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download KoBERT weights and vocabulary.')
    parser.add_argument('--cache_dir', type=str, default='.kobert/')
    args = parser.parse_args()

    model_info = kobert_models['pytorch_kobert']
    model_path = download(
        url=model_info['url'],
        filename=model_info['fname'],
        chksum=model_info['chksum'],
        cachedir=args.cache_dir,
    )
    print(f"Downloaded model to `{model_path}`")

    vocab_info = kobert_models['vocab']
    vocab_path = download(
        url=vocab_info['url'],
        filename=vocab_info['fname'],
        chksum=vocab_info['chksum'],
        cachedir=args.cache_dir,
    )
    print(f"Downloaded vocab to `{vocab_path}`")

    # Preprocess vocabulary file (_ -> ##)
    with open(vocab_path, 'rt') as f:
        vocab = json.load(f)
    print("Loaded vocabulary file.")

    vocab_file_huggingface = os.path.join(args.cache_dir, 'kobert_vocab_huggingface_format.txt')
    with open(vocab_file_huggingface, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(vocab.get('idx_to_token')))
    
    tokens = []
    for token in vocab['idx_to_token']:
        if token == '▁':
            tokens.append(token)
        else:
            token = token.replace('▁', '##')
            tokens.append(token)

    new_vocab_file_huggingface = os.path.join(args.cache_dir, 'new_kobert_vocab_huggingface_format.txt')
    with open(new_vocab_file_huggingface, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(tokens))
    print("Wrote vocab for huggingface's BertTokenizer class.")
    