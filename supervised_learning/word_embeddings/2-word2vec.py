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
        min_count: minimum number of occurrences of a word for use in training
        window: maximum distance between current and predicted word
        negative: size of negative sampling
        cbow: boolean to determine training type
        epochs: number of iterations to train over
        seed: seed for the random number generator
        workers: number of worker threads to train the model

    Returns:
        the trained model
    """
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=cbow,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    return model
