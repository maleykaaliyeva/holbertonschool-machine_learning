#!/usr/bin/env python3
"""
Dataset class for Machine Translation
"""
import transformers
from setup import load_pt2en


class Dataset:
    """
    Loads and pre-processes a dataset for machine translation
    """

    def __init__(self):
        """
        Class constructor
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for the dataset

        Args:
            data: tf.data.Dataset whose examples are formatted as (pt, en)

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            use_fast=True
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            use_fast=True
        )

        def pt_generator():
            """Yields Portuguese text in batches"""
            for pt_batch, _ in data.batch(1000):
                for pt in pt_batch:
                    yield pt.numpy().decode('utf-8')

        def en_generator():
            """Yields English text in batches"""
            for _, en_batch in data.batch(1000):
                for en in en_batch:
                    yield en.numpy().decode('utf-8')

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_generator(),
            vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_generator(),
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
