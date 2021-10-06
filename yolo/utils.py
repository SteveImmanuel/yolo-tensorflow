import tensorflow as tf
import numpy as np
import cv2

from typing import Tuple, List, Union, Iterable
from nptyping import NDArray


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


def argmax_to_max(arr, argmax, axis):
    """
    Index n dimensional array with n-1 dimensional array. Dimension of the result will be reduced
    by one.

    argmax_to_max(arr, arr.argmax(axis), axis) == arr.max(axis)
    taken from https://stackoverflow.com/questions/46103044/index-n-dimensional-array-with-n-1-d-array
    """
    new_shape = list(arr.shape)
    del new_shape[axis]

    grid = np.ogrid[tuple(map(slice, new_shape))]
    grid.insert(axis, argmax)

    return arr[tuple(grid)]


def resize_image_and_bbox(img_arr: NDArray, bbox_arr: Iterable[Iterable[int]], new_dim: Iterable) -> Tuple[NDArray, NDArray]:
    """Resize image alongside its bounding boxes

    Args:
        img_arr (NDArray): 
        bbox_arr (Iterable[Iterable[int]]): array of x, y
        new_dim (NDArray): widht, height

    Returns:
        Tuple[NDArray, NDArray]: resized img_arr and bounding boxes
    """
    ori_shape = img_arr.shape
    mat = np.array([[new_dim[0] / ori_shape[1], 0], [0, new_dim[1] / ori_shape[0]]], dtype=np.float32)
    new_bbox = np.array(list(map(lambda x: np.matmul(mat, x).astype(int), bbox_arr)))
    new_img_arr = cv2.resize(img_arr, new_dim)
    return new_img_arr, new_bbox


def bbox_mid_point_to_coordinate(bbox: Iterable[int]) -> Iterable[int]:
    """Convert x,y,w,h format to xmin,ymin,xmax,ymax

    Args:
        bbox (Iterable[int])

    Returns:
        Iterable[int]
    """
    x, y, w, h = bbox
    return x - w // 2, y - h // 2, x + w // 2, y + h // 2


def bbox_coordinate_to_mid_point(bbox: Iterable[int]) -> Iterable[int]:
    """Convert xmin,ymin,xmax,ymax format to x,y,w,h 

    Args:
        bbox (Iterable[int])
    Returns:
        Iterable[int]
    """
    xmin, ymin, xmax, ymax = bbox
    return (xmin + xmax) // 2, (ymin + ymax) // 2, (xmax - xmin), (ymax - ymin)


def normalize_bbox(bbox: NDArray, cell_width: float, cell_height: float) -> Tuple[int, int, NDArray]:
    """Convert bounding box coordinate in normalized grid

    Args:
        bbox (NDArray): C, x, y, w, h
        cell_width (float)
        cell_height (float)

    Returns:
        Tuple[int, int, NDArray]: row, col, bbox
    """
    col = int(bbox[1] // cell_width)
    row = int(bbox[2] // cell_height)
    bbox[1] = (bbox[1] % cell_width) / cell_width
    bbox[2] = (bbox[2] % cell_height) / cell_height
    bbox[3] = bbox[3] / cell_width
    bbox[4] = bbox[4] / cell_height
    return row, col, bbox


def unpack_bbox(row: int, col: int, cell_width: float, cell_height: float, bbox: NDArray) -> NDArray:
    """Convert normalized bounding box into original coordinate

    Args:
        row (int)
        col (int)
        cell_width (float)
        cell_height (float)
        bbox (NDArray): C, x, y, w, h

    Returns:
        NDArray: bbox
    """
    offset_x = col * cell_width
    offset_y = row * cell_height
    bbox[1] = bbox[1] * cell_width + offset_x
    bbox[2] = bbox[2] * cell_height + offset_y
    bbox[3] = bbox[3] * cell_width
    bbox[4] = bbox[4] * cell_height
    bbox = bbox.astype(int)
    return bbox


if __name__ == '__main__':
    assert calculate_iou(tf.constant([[.25, .25, .5, .5]]), tf.constant([[.75, .75, .5, .5]])).numpy() == 0