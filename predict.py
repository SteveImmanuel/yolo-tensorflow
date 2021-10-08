import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from yolo.v1.losses import YoloV1Loss
from yolo.visualization import decode_img
from yolo.pascal_voc_dataset import label_dict

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Predict Bounding Boxes using YOLOv1 Algorithm')
    parser.add_argument('--img-path', help='Path to image to predict', required=True)
    parser.add_argument('--model-path', help='Path to saved model', required=True)

    args = parser.parse_args()
    img_path = args.img_path
    model_path = args.model_path

    # Load model
    loss = YoloV1Loss()
    model = load_model(model_path, custom_objects={'total_loss': loss.total_loss})

    # Load image
    img = cv2.imread(img_path)  # model is trained on BGR format
    preprocessed_img = cv2.resize(img, (448, 448))
    preprocessed_img = (preprocessed_img / 255.0 - 0.5) * 2.0  # normalize to [-1, 1]
    preprocessed_img = np.array(preprocessed_img)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

    # Predict
    pred = model.predict(preprocessed_img)[0]
    
    # Show result
    decoded_img = decode_img(preprocessed_img[0], pred, {v: k for k, v in label_dict.items()})
    decoded_img = cv2.resize(decoded_img, (img.shape[1], img.shape[0]))
    cv2.imshow('YoloV1 Result', decoded_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# python predict.py --model-path=checkpoints/v1-fast/end_train --img-path=a.jpg 