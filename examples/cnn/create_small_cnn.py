# coding: utf-8

"""
Test script to create a simple DNN model.
Best to be executed in a CMSSW environment with TensorFlow and cmsml installed.
Will install tf2onnx for the user if not present in environment.

Signature: f32(64) -> f32(8)
"""

import os
import subprocess
import importlib.util

import cmsml


def create_model(
    model_dir: str,
    postfix: str = r"l{n_layers}k{kernel_size}f{n_filters}",
    n_in_1: int = 28,
    n_in_2: int = 28,
    n_out: int = 8,
    n_layers: int = 1,
    kernel_size: int = 3,
    n_filters: int = 4,
    batch_norm: bool = True,
    pooling: bool = True,
) -> None:
    # get tensorflow
    tf, _, tf_version = cmsml.tensorflow.import_tf()
    print("creating simple cnn model")
    print(f"location  : {model_dir}")
    print(f"TF version: {'.'.join(map(str, tf_version))}")

    # set random seeds to get deterministic results for testing
    tf.keras.utils.set_random_seed(1)

    # define input layer
    x = tf.keras.Input(shape=(n_in_1, n_in_2, 1), dtype=tf.float32, name="input")

    # model layers
    a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(x) if batch_norm else x
    for _ in range(n_layers):
        a = tf.keras.layers.Conv2D(n_filters, kernel_size, padding="same", activation="elu")(a)
        if pooling:
            a = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(a)
        if batch_norm:
            a = tf.keras.layers.BatchNormalization(axis=1, renorm=True)(a)

    # output layer
    b = tf.keras.layers.Flatten()(a)
    y = tf.keras.layers.Dense(n_out, activation="softmax", name="output", dtype=tf.float32)(b)

    # define the model
    model = tf.keras.Model(inputs=[x], outputs=[y])

    # test evaluation
    print(model([tf.constant([[[[i] for i in range(n_in_2)] for _ in range(n_in_1)]], dtype=tf.float32)]))

    # save it as a frozen graph
    _postfix = postfix.format(n_in_1=n_in_1, n_in_2=n_in_2, n_out=n_out, n_layers=n_layers, kernel_size=kernel_size,
                              n_filters=n_filters, batch_norm=batch_norm)
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

    # convert SavedModel to onnx
    # install tf2onnx if necessary
    if importlib.util.find_spec("tf2onnx") is None:
        subprocess.run("pip3 install --user tf2onnx", shell=True)

    # convert
    subprocess.run(
        f"""
            python3 -m tf2onnx.convert \
            --saved-model saved_model_{_postfix} \
            --output onnx_graph_{_postfix}.onnx \
        """,
        shell=True,
    )


def main():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    create_model(this_dir, n_layers=1, kernel_size=1)
    create_model(this_dir, n_layers=1, kernel_size=3)
    create_model(this_dir, n_layers=5, kernel_size=1)
    create_model(this_dir, n_layers=5, kernel_size=3)


if __name__ == "__main__":
    main()
