import argparse
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
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
    parser.add_argument('--val-img-dir', help='Directory path to validation images', required=True)
    parser.add_argument('--batch-size', help='Batch size for training', default=16, type=int)
    parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
    parser.add_argument('--load-pretrained', help='Load latest checkpoint', action='store_true')

    args = parser.parse_args()
    train_annot_dir = args.train_annot_dir
    train_img_dir = args.train_img_dir
    val_annot_dir = args.val_annot_dir
    val_img_dir = args.val_img_dir
    batch_size = args.batch_size
    load_pretrained = args.load_pretrained
    epoch = args.epoch

    model = FastYoloV1Model()
    loss = YoloV1Loss()
    model.build(input_shape=(None, 448, 448, 3))
    model.call(Input(shape=(448, 448, 3)))
    model.summary()
    model.compile(optimizer=Adam(LEARNING_RATE), loss=loss.total_loss, run_eagerly=True)

    ckpt_path = 'checkpoints/v1-fast/cp-{epoch:04d}.ckpt'
    ckpt_cb = ModelCheckpoint(ckpt_path, monitor='val_loss', mode='min', verbose=1)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=3, verbose=1, mode='min')
    tensorboard_cb = TensorBoard(log_dir='logs/v1-fast', histogram_freq=0, write_graph=True, update_freq=50)

    if load_pretrained:
        latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
        model = tf.keras.models.load_model(latest_ckpt, custom_objects={'total_loss': loss.total_loss})

    train_dataset = PascalVOCDataset(train_img_dir, train_annot_dir, batch_size)
    val_dataset = PascalVOCDataset(val_img_dir, val_annot_dir, batch_size)

    model.fit(train_dataset, validation_data=val_dataset, epochs=epoch, callbacks=[ckpt_cb, reduce_lr_cb, tensorboard_cb])
    model.save(os.path.join(os.path.dirname(ckpt_path), 'end_train'))