#!/usr/bin/env python3
"""
Dataset class for Machine Translation
"""
import tensorflow as tf
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

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

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

    def encode(self, pt, en):
        """
        Encodes a translation pair into tokens with start and end tokens

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the corresponding English sentence

        Returns:
            pt_tokens: list containing the Portuguese tokens
            en_tokens: list containing the English tokens
        """
        pt_text = pt.numpy().decode('utf-8')
        en_text = en.numpy().decode('utf-8')

        pt_tokens = self.tokenizer_pt.encode(pt_text, add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_text, add_special_tokens=False)

        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = [pt_vocab_size] + pt_tokens + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_tokens + [en_vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method

        Args:
            pt: tf.Tensor containing the Portuguese sentence
            en: tf.Tensor containing the English sentence

        Returns:
            pt_tokens: tf.Tensor containing Portuguese tokens
            en_tokens: tf.Tensor containing English tokens
        """
        pt_lang, en_lang = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_lang.set_shape([None])
        en_lang.set_shape([None])

        return pt_lang, en_lang
