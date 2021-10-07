import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import Sequence
from typing import Dict
from yolo.visualization import decode_img


class WriteImages(Callback):
    def __init__(
        self,
        logdir: str,
        dataset: Sequence,
        label_dict: Dict,
        frequency: int = 1,
        min_confidence: float = 0.75,
        max_outputs: int = 10
    ) -> None:
        super(WriteImages, self).__init__()
        self.writer = tf.summary.create_file_writer(logdir)
        self.dataset = dataset
        self.label_dict = label_dict
        self.frequency = frequency
        self.min_confidence = min_confidence
        self.max_outputs = max_outputs

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.frequency != 0:
            return

        X, _ = self.dataset[0]
        y = self.model.predict(X)
        result = []
        for image, output in zip(X, y):
            result.append(decode_img(
                image,
                output,
                min_confidence=self.min_confidence,
                label_dict=self.label_dict,
            ))
        result = np.array(result)
        with self.writer.as_default():
            tf.summary.image(
                f'Object Detection Epoch-{epoch}',
                result,
                max_outputs=self.max_outputs,
            )
