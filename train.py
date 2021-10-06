import argparse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Input
from yolo.v1.config import LEARNING_RATE
from yolo.v1.model import FastYoloV1Model
from yolo.v1.losses import YoloV1Loss
from yolo.pascal_voc_dataset import PascalVOCDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv1 Implementation using Tensorflow 2')
    parser.add_argument('--train-annot-dir', help='Directory path to training annotation', required=True)
    parser.add_argument('--train-img-dir', help='Directory path to training images', required=True)
    parser.add_argument('--val-annot-dir', help='Directory path to validation annotation', required=True)
    parser.add_argument('--val-annot-dir', help='Directory path to validation images', required=True)
    parser.add_argument('--batch-size', help='Batch size for training', default=16, type=int)
    parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
    
    args = parser.parse_args()
    train_annot_dir = args.train_annot_dir
    train_img_dir = args.train_img_dir
    val_annot_dir = args.val_annot_dir
    val_img_dir = args.val_img_dir
    batch_size = args.batch_size
    epoch = args.epoch
    
    model = FastYoloV1Model()
    loss = YoloV1Loss()
    model.build(input_shape=(None, 448, 448, 3))
    model.call(Input(shape=(448, 448, 3)))
    model.summary()
    model.compile(optimizer=Adam(LEARNING_RATE), loss=loss.total_loss, run_eagerly=True)

    ckpt_cb = ModelCheckpoint('checkpoints/v1-fast', monitor='val_loss', mode='min', save_best_only=True)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, verbose=1, mode='min')

    train_dataset = PascalVOCDataset(train_img_dir, train_annot_dir, batch_size)
    val_dataset = PascalVOCDataset(val_img_dir, val_annot_dir, batch_size=16)

    model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[ckpt_cb, reduce_lr_cb])