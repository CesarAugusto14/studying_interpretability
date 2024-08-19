from pathlib import Path


import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_all_sentences(ds, lang):
    """
    Get all sentences for the given language
    """
    for item in ds:
        yield item['translation'][lang]


def get_or_build_tokenizer(config, ds, lang):
    """
    Get or build a tokenizer for the given language
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<unk>",
                                                   "<pad>",
                                                   "<sos>",
                                                   "<eos>"],
                                   min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    """
    Get the dataset
    """
    ds_raw = load_dataset('opus_books', f'{config["lang_src"]}-{config["lang_tgt"]}')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Keep 90% of the data for training, 10% for validation
    train_ds_size = int(len(ds_raw) * 0.9)
    val_ds_size = len(ds_raw) - train_ds_size
    
    # Split the dataset
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    
    