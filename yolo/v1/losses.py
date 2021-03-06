import tensorflow as tf
import numpy as np
from yolo.utils import calculate_iou, argmax_to_max


class YoloV1Loss():
    def __init__(self, lambda_coord: int = 5, lambda_noobj: int = .5, C: int = 20, B: int = 2) -> None:
        """Initialize custom loss function

        Args:
            lambda_coord (int, optional): Defaults to 5.
            lambda_noobj (int, optional): Defaults to .5.
            C (int, optional): Total class. Defaults to 20.
            B (int, optional): Total bounding boxes per cell. Defaults to 2.
        """
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.C = C
        self.B = B

    def get_responsible_bbox(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Get index of bounding box with highest IoU with ground truth

        Args:
            y_true (tf.Tensor): (BATCH_SIZE, S, S, 5)
            y_pred (tf.Tensor): (BATCH_SIZE, S, S, B, 5)

        Returns:
            tf.Tensor: index of bounding box (BATCH_SIZE, S, S, B, 5)
        """
        iou_candidates = []
        for i in range(y_pred.shape[3]):
            iou_candidates.append(calculate_iou(y_true, y_pred[:, :, :, i, :]))
        iou_candidates = tf.convert_to_tensor(iou_candidates)
        bbox_indexes = tf.math.argmax(iou_candidates, axis=0)

        bbox_indexes = tf.one_hot(bbox_indexes, depth=self.B)
        bbox_indexes = tf.reshape(bbox_indexes, [tf.shape(bbox_indexes)[0], tf.shape(bbox_indexes)[1], tf.shape(bbox_indexes)[2], self.B])
        bbox_indexes = tf.repeat(bbox_indexes, repeats=5, axis=3)  # 5 from C, x, y, w, h
        bbox_indexes = tf.reshape(bbox_indexes, tf.shape(y_pred))
        return bbox_indexes

    def bbox_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate bounding box loss. Only check the bounding box with highest IoU

        Args:
            y_true (tf.Tensor): (BATCH_SIZE, S, S, 5) 5-> C, x, y, w, h
            y_pred (tf.Tensor): (BATCH_SIZE, S, S, 5)

        Returns:
            tf.Tensor: (1,)
        """
        xy_loss = tf.math.squared_difference(y_true[..., 1:3], y_pred[..., 1:3])
        wh_loss = tf.math.squared_difference(
            tf.math.sqrt(y_true[..., 3:]),
            tf.math.sign(y_pred[..., 3:]) * tf.math.sqrt(tf.math.abs(y_pred[..., 3:]))
        )

        return tf.math.reduce_mean(tf.math.reduce_sum(xy_loss + wh_loss, axis=(1, 2, 3)))

    def anchor_box_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, is_object_exist: tf.Tensor) -> tf.Tensor:
        """Calculate bbox and object loss. Only check the bounding box with highest IoU

        Args:
            y_true (tf.Tensor): (BATCH_SIZE, S, S, 5*B)
            y_pred (tf.Tensor): (BATCH_SIZE, S, S, 5*B)
            is_object_exist (tf.Tensor): (BATCH_SIZE, S, S, 1)

        Returns:
            tf.Tensor
        """
        gtruth_bbox = y_true[..., 0:5]
        pred_unpack = tf.reshape(y_pred, [tf.shape(y_pred)[0], tf.shape(y_pred)[1], tf.shape(y_pred)[2], self.B, 5])
        bbox_indexes = self.get_responsible_bbox(gtruth_bbox, pred_unpack)

        gtruth_bbox *= is_object_exist  # (BATCH_SIZE, S, S, 5)
        pred_bbox = is_object_exist * tf.math.reduce_sum(bbox_indexes * pred_unpack, axis=3)

        bbox_loss = self.bbox_loss(gtruth_bbox, pred_bbox)
        object_loss = tf.math.reduce_mean(
            tf.math.reduce_sum(tf.math.squared_difference(gtruth_bbox[..., 0], pred_bbox[..., 0]), axis=(1, 2))
        )

        no_object_loss = tf.zeros(
            (tf.shape(is_object_exist)[0], tf.shape(is_object_exist)[1], tf.shape(is_object_exist)[2]), dtype=tf.float32
        )
        for i in range(self.B):
            no_object_loss += (1 - is_object_exist[..., 0]) * tf.math.squared_difference(gtruth_bbox[..., 0], pred_unpack[:, :, :, i, 0])

        no_object_loss = tf.math.reduce_mean(tf.math.reduce_sum(no_object_loss, axis=(1, 2)))

        return bbox_loss * self.lambda_coord + object_loss + no_object_loss * self.lambda_noobj

    def class_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, is_object_exist: tf.Tensor) -> tf.Tensor:
        """Calculate class probability loss

        Args:
            y_true (tf.Tensor): (BATCH_SIZE, S, S, C)
            y_pred (tf.Tensor): (BATCH_SIZE, S, S, C)
            is_object_exist (tf.Tensor): (BATCH_SIZE, S, S, 1)

        Returns:
            tf.Tensor
        """
        return tf.math.reduce_mean(tf.math.reduce_sum(is_object_exist * tf.math.squared_difference(y_true, y_pred), axis=(1, 2, 3)))

    def total_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate total loss from YoloV1 Paper

        Args:
            y_true (tf.Tensor): (BATCH_SIZE, S, S, C+5*B)
            y_pred (tf.Tensor): (BATCH_SIZE, S, S, C+5*B)

        Returns:
            tf.Tensor
        """
        is_object_exist = y_true[..., 20:21]  # (BATCH_SIZE, S, S, 1)
        class_loss = self.class_loss(y_true[..., :self.C], y_pred[..., :self.C], is_object_exist)
        anchor_box_loss = self.anchor_box_loss(y_true[..., self.C:], y_pred[..., self.C:], is_object_exist)
        return class_loss + anchor_box_loss


if __name__ == '__main__':
    y_true = np.zeros((64, 7, 7, 30))
    y_pred = np.ones((64, 7, 7, 30))
    loss = YoloV1Loss(C=20, B=2)
    res = loss.total_loss(y_true, y_pred)
    print(res)
