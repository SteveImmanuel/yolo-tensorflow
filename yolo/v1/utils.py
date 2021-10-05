import tensorflow as tf
from typing import List

@tf.function(experimental_follow_type_hints=True)
def calculate_iou(bbox1: tf.Tensor, bbox2: tf.Tensor) -> tf.Tensor:
    """Calculate intersection over union between bounding boxes.
    bbox shape = (BATCH_SIZE, S, S, 4)
    4 -> x, y, w, h
    
    Args:
        bbox1 (tf.Tensor) 
        bbox2 (tf.Tensor) 

    Returns:
        tf.Tensor: (BATCH_SIZE, S, S)
    """
    area_bbox1 = bbox1[..., 2] * bbox1[..., 3]
    area_bbox2 = bbox2[..., 2] * bbox2[..., 3]

    bbox1_x1 = bbox1[..., 0] - bbox1[..., 2] / 2
    bbox1_y1 = bbox1[..., 1] - bbox1[..., 3] / 2
    bbox1_x2 = bbox1[..., 0] + bbox1[..., 2] / 2
    bbox1_y2 = bbox1[..., 1] + bbox1[..., 3] / 2
    bbox2_x1 = bbox2[..., 0] - bbox2[..., 2] / 2
    bbox2_y1 = bbox2[..., 1] - bbox2[..., 3] / 2
    bbox2_x2 = bbox2[..., 0] + bbox2[..., 2] / 2
    bbox2_y2 = bbox2[..., 1] + bbox2[..., 3] / 2

    inter_box_x1 = tf.math.maximum(bbox1_x1, bbox2_x1)
    inter_box_y1 = tf.math.maximum(bbox1_y1, bbox2_y1)
    inter_box_x2 = tf.math.minimum(bbox1_x2, bbox2_x2)
    inter_box_y2 = tf.math.minimum(bbox1_y2, bbox2_y2)

    inter_w = tf.nn.relu(inter_box_x2 - inter_box_x1)
    inter_h = tf.nn.relu(inter_box_y2 - inter_box_y1)
    area_inter = inter_w * inter_h

    return area_inter / (area_bbox1 + area_bbox2 - area_inter + 1e-6)


if __name__ == '__main__':
    assert calculate_iou(tf.constant([[.25, .25, .5, .5]]), tf.constant([[.75, .75, .5, .5]])).numpy() == 0