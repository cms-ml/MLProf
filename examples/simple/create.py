# coding: utf-8

"""
Test script to create a simple model for AOT compilation tests.

Signature: f32(4) -> f32(2)
"""

import os

import cmsml


def create_model(model_dir):
    # get tensorflow (suppressing the usual device warnings and logs)
    tf, _, tf_version = cmsml.tensorflow.import_tf()
    print("creating simple model")
    print(f"location  : {model_dir}")
    print(f"TF version: {'.'.join(map(str, tf_version))}")

    # set random seeds to get deterministic results for testing
    tf.keras.utils.set_random_seed(1)

    # define architecture
    n_in, n_out, n_layers, n_units = 4, 2, 5, 128

    # define input layer
    x = tf.keras.Input(shape=(n_in,), dtype=tf.float32, name="input")

    # model layers
    a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(x)
    for _ in range(n_layers):
        a = tf.keras.layers.Dense(n_units, activation="tanh")(a)
        a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(a)

    # output layer
    y = tf.keras.layers.Dense(n_out, activation="softmax", name="output", dtype=tf.float32)(a)

    # define the model
    model = tf.keras.Model(inputs=[x], outputs=[y])

    # test evaluation
    print(model([tf.constant([list(range(n_in))], dtype=tf.float32)]))

    # save it
    tf.saved_model.save(model, model_dir)


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    create_model(os.path.join(this_dir, "saved_model"))


if __name__ == "__main__":
    main()
