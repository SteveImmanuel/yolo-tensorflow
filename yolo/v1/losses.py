import tensorflow as tf
from yolo.v1.utils import calculate_iou


class YoloV1Loss():
    def __init__(self, lambda_coord: int = 5, lambda_noobj: int = .5, C: int = 20, B: int = 2) -> None:
        """Initialize custom loss function

        Args:
            lambda_coord (int, optional): Defaults to 5.
            lambda_noobj (int, optional): Defaults to .5.
            C (int, optional): [description]. Total class. Defaults to 20.
            B (int, optional): [description]. Total bounding boxes per cell. Defaults to 2.
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
            tf.Tensor: index of bounding box (BATCH, S, S)
        """
        iou_candidates = []
        for i in range(y_pred.shape[3]):
            iou_candidates.append(calculate_iou(y_true, y_pred[:, :, :, i, :]))
        iou_candidates = tf.convert_to_tensor(iou_candidates)
        bbox_indexes = tf.math.argmax(iou_candidates, axis=0)
        return bbox_indexes

    def bbox_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate bounding box loss. Only check the bounding box with highest IoU

        Args:
            y_true (tf.Tensor): (BATCH_SIZE, S, S, 5*B)
            y_pred (tf.Tensor): (BATCH_SIZE, S, S, 5*B)

        Returns:
            tf.Tensor: [description]
        """
        pred_unpack = tf.reshape(y_pred, (*y_pred.shape[:-1], -1, 5))
        bbox_indexes = self.get_responsible_bbox(y_true[..., self.C:self.C + 5], pred_unpack)

    def object_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate object loss

        Args:
            y_true (tf.Tensor): [description]
            y_pred (tf.Tensor): [description]

        Returns:
            tf.Tensor: [description]
        """

    def class_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Calculate class probability loss

        Args:
            y_true (tf.Tensor): (BATCH_SIZE, S, S, C)
            y_pred (tf.Tensor): (BATCH_SIZE, S, S, C)

        Returns:
            tf.Tensor
        """
        return tf.math.reduce_sum(tf.math.squared_difference(y_true, y_pred))