#!/usr/bin/env python3
"""Transfer learning for CIFAR-10."""

from tensorflow import keras as K


def preprocess_data(X, Y):
    """Preprocess CIFAR-10 data."""
    X_p = X.astype('float32') / 255.0
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p


if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = (
        K.datasets.cifar10.load_data()
    )

    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = K.Input(shape=(32, 32, 3))

    resize = K.layers.Lambda(
        lambda x: K.ops.image.resize(x, (224, 224))
    )(inputs)

    base_model = K.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=resize,
        pooling='avg'
    )

    base_model.trainable = False

    train_features = base_model.predict(
        X_train,
        batch_size=128,
        verbose=1
    )

    test_features = base_model.predict(
        X_test,
        batch_size=128,
        verbose=1
    )

    feature_input = K.Input(
        shape=train_features.shape[1:]
    )

    x = K.layers.Dense(
        512,
        activation='relu'
    )(feature_input)

    x = K.layers.Dropout(0.4)(x)

    x = K.layers.Dense(
        256,
        activation='relu'
    )(x)

    x = K.layers.Dropout(0.3)(x)

    outputs = K.layers.Dense(
        10,
        activation='softmax'
    )(x)

    classifier = K.models.Model(
        inputs=feature_input,
        outputs=outputs
    )

    classifier.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=0.001
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    classifier.fit(
        train_features,
        Y_train,
        batch_size=128,
        epochs=30,
        validation_data=(test_features, Y_test),
        verbose=1
    )

    final_inputs = K.Input(shape=(32, 32, 3))

    final_resize = K.layers.Lambda(
        lambda x: K.ops.image.resize(x, (224, 224))
    )(final_inputs)

    final_base = K.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=final_resize,
        pooling='avg'
    )

    final_base.trainable = False

    x = final_base.output

    x = K.layers.Dense(
        512,
        activation='relu'
    )(x)

    x = K.layers.Dropout(0.4)(x)

    x = K.layers.Dense(
        256,
        activation='relu'
    )(x)

    x = K.layers.Dropout(0.3)(x)

    final_outputs = K.layers.Dense(
        10,
        activation='softmax'
    )(x)

    model = K.models.Model(
        inputs=final_inputs,
        outputs=final_outputs
    )

    model.compile(
        optimizer=K.optimizers.Adam(
            learning_rate=0.001
        ),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.get_layer(
        'efficientnetb0'
    ).set_weights(final_base.get_weights())

    classifier_weights = classifier.get_weights()

    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, K.layers.Dense)
    ]

    for layer, weights in zip(dense_layers, [
            classifier_weights[0],
            classifier_weights[1],
            classifier_weights[2],
            classifier_weights[3],
            classifier_weights[4],
            classifier_weights[5]]):
        layer.set_weights(weights)

    model.save('cifar10.h5')
