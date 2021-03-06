import cv2
import os
import numpy as np
import xml.etree.ElementTree as ET

from typing import Tuple
from nptyping import NDArray
from tensorflow.keras.utils import Sequence
from yolo.utils import resize_image_and_bbox, bbox_coordinate_to_mid_point, normalize_bbox
from yolo.visualization import visualize_img

label_dict = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19
}


class PascalVOCDataset(Sequence):
    def __init__(
        self,
        img_dir: str,
        annotation_dir: str,
        batch_size: int = 64,
        input_dim: Tuple[int, int] = (448, 448),
        S: int = 7,
        B: int = 2,
        test_overfit: bool = False,
    ) -> None:
        super(PascalVOCDataset, self).__init__()
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.S = S
        self.B = B
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.test_overfit = test_overfit

        self._initialize_data()
        self.cell_width = float(self.input_dim[1]) / self.S
        self.cell_height = float(self.input_dim[0]) / self.S

    def _initialize_data(self):
        self.annot_paths = np.array(os.listdir(self.annotation_dir))
        self.total_batch = len(self.annot_paths) // self.batch_size
        self._shuffle_data()

    def _shuffle_data(self):
        random_ids = np.random.permutation(len(self.annot_paths))
        self.annot_paths = self.annot_paths[random_ids]

    def _parse_annotation(self, annot_path: str) -> Tuple[NDArray, NDArray[NDArray]]:
        try:
            xml_file = open(annot_path, 'r')
            tree = ET.parse(xml_file)
            root = tree.getroot()

            img_filename = root.find('filename').text
            objects = root.findall('.//object')
            bboxes = []
            for object in objects:
                label = object.find('name').text
                bbox = object.find('bndbox')
                xmin, ymin = int(float(bbox.find('xmin').text)), int(float(bbox.find('ymin').text))
                xmax, ymax = int(float(bbox.find('xmax').text)), int(float(bbox.find('ymax').text))
                bboxes.append([label_dict[label], xmin, ymin, xmax, ymax])

            img_path = os.path.join(self.img_dir, img_filename)
            img_arr = cv2.imread(img_path)
            img_arr = (img_arr / 255.0 - 0.5) * 2

            xml_file.close()
            return img_arr, np.array(bboxes)
        except:
            print('error xml parse', annot_path)

    def _preprocess_img(self, img_arr: NDArray, bboxes: NDArray[NDArray]) -> Tuple[NDArray, NDArray[NDArray]]:
        bboxes_coord = bboxes[:, 1:].reshape(-1, 2)
        resized_img_arr, resized_bbox_coord = resize_image_and_bbox(img_arr, bboxes_coord, self.input_dim)
        resized_bbox_coord = resized_bbox_coord.reshape(-1, 4)
        bboxes[:, 1:] = resized_bbox_coord

        return resized_img_arr, bboxes

    def _create_target(self, bboxes: NDArray[NDArray]) -> NDArray:
        target = np.zeros((self.S, self.S, 20 + self.B * 5), dtype=np.float32)

        bboxes = bboxes.astype(float)
        bboxes[:, 1:] = list(map(lambda x: bbox_coordinate_to_mid_point(x[1:]), bboxes))  # TOTAL_BBOX, 5

        for bbox in bboxes:
            row, col, bbox = normalize_bbox(bbox, self.cell_width, self.cell_height)
            target[row][col][20] = 1
            target[row][col][21:25] = bbox[1:]
            target[row][col][int(bbox[0])] = 1

        return target

    def _generate_batch(self, index):
        selected_annot_paths = self.annot_paths[index * self.batch_size:(index + 1) * self.batch_size]
        X = []
        y = []
        for path in selected_annot_paths:
            full_path = os.path.join(self.annotation_dir, path)
            img_arr, bboxes = self._preprocess_img(*self._parse_annotation(full_path))
            X.append(img_arr)
            y.append(self._create_target(bboxes))
        return np.array(X), np.array(y)

    def __len__(self):
        if self.test_overfit:
            return 2
        return self.total_batch

    def __getitem__(self, index):
        return self._generate_batch(index)

    def on_epoch_end(self):
        if not self.test_overfit:
            self._shuffle_data()


if __name__ == '__main__':
    dataset = PascalVOCDataset('dataset/VOC2012_train/JPEGImages', 'dataset/VOC2012_train/Annotations')
    X, y = dataset[0]
    print(X.shape, y.shape)
    visualize_img(X[0], y[0], {v: k for k, v in label_dict.items()})