#!/usr/bin/env python3
"""
Word2Vec Model Module
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a gensim word2vec model.

    Args:
        sentences: list of sentences to be trained on
        vector_size: dimensionality of the embedding layer
        min_count: minimum number of occurrences for training
        window: maximum distance between current and predicted word
        negative: size of negative sampling
        cbow: boolean; True for CBOW (sg=0), False for Skip-gram (sg=1)
        epochs: number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads

    Returns:
        the trained Word2Vec model
    """
    # Initialize model shell without sentences
    model = gensim.models.Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=0 if cbow else 1,
        seed=seed,
        workers=workers
    )

    # Build vocabulary first
    model.build_vocab(sentences)

    # Train model explicitly over specified epochs
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs
    )

    return model
