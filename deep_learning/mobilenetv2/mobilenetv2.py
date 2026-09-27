from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# MobileNetV2 doc
# https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2

"""
MobileNetV2 Trainer Class

This class provides methods to load data, build a MobileNetV2 model,
train the model, and export training results.
"""


class MobileNetV2Trainer:

    def __init__(
        self,
        data_dir,
        model_path,
        results_dir,
        image_size=(224, 224),
        batch_size=32,
        epochs=10,
        learning_rate=0.0001,
        trainable=True,
    ):
        """
        Initialize model attributes related to data, model, and training configuration.
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.model_path = Path(model_path)

        self.image_size = image_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.trainable = trainable

        self.model = None
        self.history = None

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_data(self):
        """
        load all datasets (train, validation, test) from the specified data directory.
        """

        train_dir = self.data_dir / "train"
        val_dir = self.data_dir / "val"
        test_dir = self.data_dir / "test"

        self.train_dataset = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="binary",
            shuffle=True,
        )

        self.val_dataset = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="binary",
            shuffle=False,
        )

        self.test_dataset = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            label_mode="binary",
            shuffle=False,
        )

        print("Classes:", self.train_dataset.class_names)

        autotune = tf.data.AUTOTUNE

        self.train_dataset = self.train_dataset.prefetch(autotune)
        self.val_dataset = self.val_dataset.prefetch(autotune)
        self.test_dataset = self.test_dataset.prefetch(autotune)

    def build_model(self):
        """
        Build MobileNetV2 model with specified configuration.
        """

        base_model = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(
                self.image_size[0],
                self.image_size[1],
                3,
            ),
        )

        print("MobileNetV2 loaded successfully!")
        print("TensorFlow version:", tf.__version__)
        print("GPU devices:", tf.config.list_physical_devices("GPU"))
        print("Input shape:", base_model.input_shape)
        print("Output shape:", base_model.output_shape)

        base_model.trainable = self.trainable

        inputs = layers.Input(
            shape=(
                self.image_size[0],
                self.image_size[1],
                3,
            )
        )

        x = preprocess_input(inputs)

        x = base_model(x, training=False)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(1, activation="sigmoid")(x)

        self.model = models.Model(inputs, outputs)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        self.model.summary()

    def train(self):
        """
        Train MobileNetV2 model using the loaded datasets.
        """

        self.history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=self.epochs,
        )

        return self.history

    def export_training_results(self):
        """
        Export training results, including accuracy and loss plots, and save results to directory.
        """

        if self.history is None:
            print("No training history available.")
            return

        self.results_dir.mkdir(parents=True, exist_ok=True)

        history = self.history.history

        history_df = pd.DataFrame(history)
        history_df.index += 1
        history_df.index.name = "epoch"

        history_df.to_csv(self.results_dir / "training_history.csv")

        plt.figure()

        plt.plot(history["accuracy"], label="Training Accuracy")

        plt.plot(history["val_accuracy"], label="Validation Accuracy")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()

        plt.savefig(self.results_dir / "accuracy.png")

        plt.close()

        plt.figure()

        plt.plot(history["loss"], label="Training Loss")

        plt.plot(history["val_loss"], label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()

        plt.savefig(self.results_dir / "loss.png")

        plt.close()

        print("Training results exported successfully!")

    def evaluate(self):
        """
        Evaluate MobileNetV2 model on test dataset.
        """

        test_loss, test_accuracy = self.model.evaluate(self.test_dataset)

        print("Test loss:", test_loss)
        print("Test accuracy:", test_accuracy)

        return test_loss, test_accuracy

    def save(self):
        """
        Save trained MobileNetV2 model to specified model path.
        """

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(self.model_path)

        print("Model saved successfully!")
        print("Model path:", self.model_path)
