# coding: utf-8

"""
Test script to create a simple CNN model.
Best to be executed in a CMSSW environment with TensorFlow and cmsml installed or a dedicated virtual environment.

Signature: f32(1024 * n_channels) -> f32(8)

1024 could correspond to a pixelated 32x32 "image" of an eta-phi plane.
"""

import os
import subprocess

import cmsml  # type: ignore[import-untyped]


def create_model(
    model_dir: str,
    postfix: str = r"i{n_in}c{n_channels}v{n_convs}l{n_layers}u{n_units}",
    n_in: int = 32,
    n_channels: int = 1,
    n_filters: int = 32,
    n_convs: int = 3,
    n_out: int = 8,
    n_layers: int = 10,
    n_units: int = 128,
) -> None:
    # get tensorflow
    tf, _, tf_version = cmsml.tensorflow.import_tf()
    print("creating simple model")
    print(f"location  : {model_dir}")
    print(f"TF version: {'.'.join(map(str, tf_version))}")

    # set random seeds to get deterministic results for testing
    tf.keras.utils.set_random_seed(1)

    # define input layer
    x = tf.keras.Input(shape=(n_channels * n_in**2,), dtype=tf.float32, name="input")

    # reshape
    a = tf.keras.layers.Reshape((n_in, n_in, n_channels))(x)

    # convolutions and pooling
    for _ in range(n_convs):
        a = tf.keras.layers.Conv2D(n_filters, (3, 3), activation="elu")(a)
        a = tf.keras.layers.MaxPooling2D((2, 2))(a)

    # flatten
    a = tf.keras.layers.Flatten()(a)

    # model layers
    for _ in range(n_layers):
        a = tf.keras.layers.Dense(n_units, activation="elu")(a)

    # output layer
    y = tf.keras.layers.Dense(n_out, activation="softmax", name="output", dtype=tf.float32)(a)

    # define the model
    model = tf.keras.Model(inputs=[x], outputs=[y])

    # test evaluation
    print(model([tf.constant([list(range(n_channels * n_in**2))], dtype=tf.float32)]))

    # save it as a frozen graph
    _postfix = postfix.format(
        n_in=n_in,
        n_out=n_out,
        n_channels=n_channels,
        n_filters=n_filters,
        n_convs=n_convs,
        n_layers=n_layers,
        n_units=n_units,
    )
    cmsml.tensorflow.save_graph(
        os.path.join(model_dir, f"frozen_graph_{_postfix}.pb"),
        model,
        variables_to_constants=True,
    )

    # create a SavedModel
    saved_model_dir = os.path.join(model_dir, f"saved_model_{_postfix}")
    tf.saved_model.save(model, saved_model_dir)

    # convert to onnx
    cmd = [
        "python",
        "-m", "tf2onnx.convert",
        "--saved-model", saved_model_dir,
        "--output", os.path.join(model_dir, f"onnx_graph_{_postfix}.onnx"),
    ]
    subprocess.run(cmd, check=True)


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))

    create_model(this_dir, n_in=32, n_channels=1, n_convs=1, n_layers=10)
    create_model(this_dir, n_in=32, n_channels=1, n_convs=3, n_layers=10)
    create_model(this_dir, n_in=64, n_channels=3, n_convs=4, n_layers=10)
    create_model(this_dir, n_in=32, n_channels=1, n_convs=3, n_layers=20)
    create_model(this_dir, n_in=64, n_channels=3, n_convs=4, n_layers=20)


if __name__ == "__main__":
    main()
