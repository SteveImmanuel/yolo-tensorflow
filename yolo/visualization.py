import cv2
import numpy as np

from nptyping import NDArray
from typing import Dict
from yolo.utils import unpack_bbox, bbox_mid_point_to_coordinate


def decode_img(img_arr: NDArray, output_matrix: NDArray, label_dict: Dict, min_confidence: float = .8, normalize: bool = True) -> NDArray:
    """Add bounding box and label to image based on output_matrix

    Args:
        img_arr (NDArray): row x col x channel
        output_matrix (NDArray): S x S x C+5*B
        label_dict (Dict)
        min_confidence (float, optional): Defaults to .8.
        normalize (bool): Map -1 to 1 range to 0 to 1 range
    """
    cell_width = img_arr.shape[1] / output_matrix.shape[1]
    cell_height = img_arr.shape[0] / output_matrix.shape[0]
    for i in range(len(output_matrix)):
        for j in range(len(output_matrix[i])):
            object_label = label_dict[np.argmax(output_matrix[i][j][:20])]
            max_confidence_idx = np.argmax(output_matrix[i][j][20::5])
            max_confidence = output_matrix[i][j][20 + 5 * max_confidence_idx]
            
            if max_confidence < min_confidence:
                continue

            bbox = output_matrix[i][j][20 + 5 * max_confidence_idx:20 + 5 * (max_confidence_idx + 1)]
            bbox = unpack_bbox(i, j, cell_width, cell_height, bbox)
            bbox = bbox_mid_point_to_coordinate(bbox[1:])
            img_arr = cv2.rectangle(img_arr, bbox[:2], bbox[2:], (255, 0, 0), 2)
            cv2.putText(img_arr, f'{object_label} {max_confidence:.2f}', bbox[:2], cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 80, 50), 2)
    if normalize:
        img_arr = (img_arr + 1) / 2
    return img_arr


def visualize_img(img_arr: NDArray, output_matrix: NDArray, label_dict: Dict, min_confidence: float = .8) -> None:
    """Visualize image with bounding boxes in new window

    Args:
        img_arr (NDArray): row x col x channel
        output_matrix (NDArray): S x S x C+5*B
        label_dict (Dict)
        min_confidence (float, optional): Defaults to .8.
    """
    img_arr = decode_img(img_arr, output_matrix, label_dict, min_confidence)
    cv2.imshow('YOLOv1 Result', img_arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
