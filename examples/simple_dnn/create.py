# coding: utf-8

"""
Test script to create a simple DNN model.
Best to be executed in a CMSSW environment with TensorFlow and cmsml installed.

Signature: f32(64) -> f32(8)
"""

import os

import cmsml


def create_model(
    model_dir: str,
    postfix: str = r"l{n_layers}u{n_units}",
    n_in: int = 64,
    n_out: int = 8,
    n_layers: int = 10,
    n_units: int = 256,
    batch_norm: bool = True,
) -> None:
    # get tensorflow
    tf, _, tf_version = cmsml.tensorflow.import_tf()
    print("creating simple model")
    print(f"location  : {model_dir}")
    print(f"TF version: {'.'.join(map(str, tf_version))}")

    # set random seeds to get deterministic results for testing
    tf.keras.utils.set_random_seed(1)

    # define input layer
    x = tf.keras.Input(shape=(n_in,), dtype=tf.float32, name="input")

    # model layers
    a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(x) if batch_norm else x
    for _ in range(n_layers):
        a = tf.keras.layers.Dense(n_units, activation="elu")(a)
        if batch_norm:
            a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(a)

    # output layer
    y = tf.keras.layers.Dense(n_out, activation="softmax", name="output", dtype=tf.float32)(a)

    # define the model
    model = tf.keras.Model(inputs=[x], outputs=[y])

    # test evaluation
    print(model([tf.constant([list(range(n_in))], dtype=tf.float32)]))

    # save it as a frozen graph
    _postfix = postfix.format(n_in=n_in, n_out=n_out, n_layers=n_layers, n_units=n_units, batch_norm=batch_norm)
    cmsml.tensorflow.save_graph(
        os.path.join(model_dir, f"frozen_graph_{_postfix}.pb"),
        model,
        variables_to_constants=True,
    )

    # create a SavedModel
    tf.saved_model.save(
        model,
        os.path.join(model_dir, f"saved_model_{_postfix}"),
    )


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    create_model(this_dir, n_layers=10, n_units=128)
    create_model(this_dir, n_layers=10, n_units=256)
    create_model(this_dir, n_layers=20, n_units=128)
    create_model(this_dir, n_layers=20, n_units=256)


if __name__ == "__main__":
    main()
