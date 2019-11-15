# -*- coding: utf-8 -*-

import os
import sys
import json
import hashlib
import argparse
import requests

from transformers import BertConfig, BertModel


HUGGINGFACE_VOCAB_FILE = "kobert_vocab_huggingface_format.txt"
NEW_HUGGINGFACE_VOCAB_FILE = "new_kobert_vocab_huggingface_format.txt"
KOBERT_CONFIG_FILE = "kobert-8002-config.json"
DISTILKOBERT_CONFIG_FILE = "distilkobert_student_config.json"
KOBERT_TOKENIZER_CONFIG_FILE = "kobert_tokenizer_config.json"


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

kobert_config = {
    'attention_probs_dropout_prob': 0.1,
    'hidden_act': 'gelu',
    'hidden_dropout_prob': 0.1,
    'hidden_size': 768,
    'initializer_range': 0.02,
    'intermediate_size': 3072,
    'max_position_embeddings': 512,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'type_vocab_size': 2,
    'vocab_size': 8002
}

distilkobert_config = {
    "activation": "gelu",
    "attention_dropout": 0.1,
    "dim": 768,
    "dropout": 0.1,
    "finetuning_task": None,
    "hidden_dim": 3072,
    "initializer_range": 0.02,
    "max_position_embeddings": 512,
    "n_heads": 12,
    "n_layers": 6,
    "num_labels": 2,
    "output_attentions": False,
    "output_hidden_states": False,
    "pruned_heads": {},
    "qa_dropout": 0.1,
    "seq_classif_dropout": 0.2,
    "sinusoidal_pos_embds": False,
    "tie_weights_": True,
    "torchscript": False,
    "vocab_size": 8002
}

kobert_tokenizer_config = {
    "vocab_file": None,
    "do_lower_case": False,
    "do_basic_tokenize": True,
    "never_split": None,
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
    "tokenize_chinese_chars": False
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

    # Download model
    model_info = kobert_models['pytorch_kobert']
    model_path = download(
        url=model_info['url'],
        filename=model_info['fname'],
        chksum=model_info['chksum'],
        cachedir=args.cache_dir,
    )
    print(f"Downloaded model to `{model_path}`")

    # Download vocabulary
    vocab_info = kobert_models['vocab']
    vocab_path = download(
        url=vocab_info['url'],
        filename=vocab_info['fname'],
        chksum=vocab_info['chksum'],
        cachedir=args.cache_dir,
    )
    print(f"Downloaded vocab to `{vocab_path}`")

    # Load vocab
    with open(vocab_path, 'rt') as f:
        vocab = json.load(f)
    print("Loaded vocabulary file.")

    # Save vocab in huggingface format
    vocab_file_huggingface = os.path.join(args.cache_dir, HUGGINGFACE_VOCAB_FILE)
    with open(vocab_file_huggingface, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(vocab.get('idx_to_token')))

    # Preprocess vocabulary file (_ -> ##)
    tokens = []
    for token in vocab['idx_to_token']:
        if token == '▁':
            tokens.append(token)
        else:
            token = token.replace('▁', '##')
            tokens.append(token)

    # Save new vocab in huggingface format
    new_vocab_file_huggingface = os.path.join(args.cache_dir, NEW_HUGGINGFACE_VOCAB_FILE)
    with open(new_vocab_file_huggingface, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(tokens))
    print("Wrote vocab for huggingface's BertTokenizer class.")

    # Save KoBERT model using `.save_pretrained`
    kobert_model = BertModel(config=BertConfig.from_dict(kobert_config))
    pretrained_dir = os.path.join(args.cache_dir, 'pretrained/')
    os.makedirs(pretrained_dir, exist_ok=True)
    kobert_model.save_pretrained(save_directory=pretrained_dir)
    print(f"Saved KoBERT model using `.save_pretrained.` to {pretrained_dir}")

    # Save KoBERT configurations
    with open(os.path.join(args.cache_dir, KOBERT_CONFIG_FILE), 'w') as f:
        json.dump(kobert_config, f, indent=4)
    print("Saved KoBERT configurations.")

    # Save DistilKoBERT configurations
    with open(os.path.join(args.cache_dir, DISTILKOBERT_CONFIG_FILE), 'w') as f:
        json.dump(distilkobert_config, f, indent=4)

    # Save KoBERT tokenizer configurations
    kobert_tokenizer_config['vocab_file'] = os.path.join(args.cache_dir, NEW_HUGGINGFACE_VOCAB_FILE)
    with open(os.path.join(args.cache_dir, KOBERT_TOKENIZER_CONFIG_FILE), 'w') as f:
        json.dump(kobert_tokenizer_config, f, indent=4)
