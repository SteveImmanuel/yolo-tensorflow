import argparse
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras import Input
from yolo.v1.model import FastYoloV1Model, YoloV1Model
from yolo.v1.losses import YoloV1Loss
from yolo.pascal_voc_dataset import PascalVOCDataset
from yolo.callbacks import WriteImages
from yolo.pascal_voc_dataset import label_dict

if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='YOLOv1 Implementation using Tensorflow 2')
    parser.add_argument('--train-annot-dir', help='Directory path to training annotation', required=True)
    parser.add_argument('--train-img-dir', help='Directory path to training images', required=True)
    parser.add_argument('--val-annot-dir', help='Directory path to validation annotation', required=True)
    parser.add_argument('--val-img-dir', help='Directory path to validation images', required=True)
    parser.add_argument('--batch-size', help='Batch size for training', default=16, type=int)
    parser.add_argument('--epoch', help='Total epoch', default=20, type=int)
    parser.add_argument('--learning-rate', help='Learning rate for training', default=1e-3, type=float)
    parser.add_argument('--test-overfit', help='Sanity check to test overfit model with very small dataset', action='store_true')
    parser.add_argument('--model-path', help='Load pretrained model')

    args = parser.parse_args()
    train_annot_dir = args.train_annot_dir
    train_img_dir = args.train_img_dir
    val_annot_dir = args.val_annot_dir
    val_img_dir = args.val_img_dir
    batch_size = args.batch_size
    epoch = args.epoch
    learning_rate = args.learning_rate
    test_overfit = args.test_overfit
    model_path = args.model_path

    # show training config
    print('TRAINING CONFIGURATION')
    print('Train annotation directory:', train_annot_dir)
    print('Train image directory:', train_img_dir)
    print('Validation annotation directory:', val_annot_dir)
    print('Validation image directory:', val_img_dir)
    print('Batch size:', batch_size)
    print('Epoch:', epoch)
    print('Learning rate:', learning_rate)
    print('Test overfit:', test_overfit)
    print('Model path:', model_path)

    # Create and compile model
    model = FastYoloV1Model()
    loss = YoloV1Loss()
    model.build(input_shape=(None, 448, 448, 3))
    model.call(Input(shape=(448, 448, 3)))
    model.summary()
    model.compile(optimizer=Adam(learning_rate, beta_1=0.5, beta_2=0.995), loss=loss.total_loss, run_eagerly=True)

    # define datasets
    train_dataset = PascalVOCDataset(train_img_dir, train_annot_dir, batch_size, test_overfit=test_overfit)
    val_dataset = PascalVOCDataset(val_img_dir, val_annot_dir, batch_size, test_overfit=test_overfit)

    # define all callbacks
    ckpt_path = 'checkpoints/v1-fast/cp-{epoch:04d}.ckpt'
    log_dir = 'logs/v1-fast'
    ckpt_cb = ModelCheckpoint(ckpt_path, monitor='val_loss', mode='min', verbose=1)
    tensorboard_cb = TensorBoard(log_dir, histogram_freq=0, write_graph=True, update_freq=50)
    early_stop_cb = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)
    if test_overfit:
        reduce_lr_cb = ReduceLROnPlateau(monitor='loss', factor=.2, patience=10, verbose=1, mode='min')
        write_images_cb = WriteImages(os.path.join(log_dir, 'images'), train_dataset, {v: k for k, v in label_dict.items()})
    else:
        reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=.2, patience=2, verbose=1, mode='min')
        write_images_cb = WriteImages(os.path.join(log_dir, 'images'), val_dataset, {v: k for k, v in label_dict.items()})

    # load pretrained model if asked
    if model_path:
        model = tf.keras.models.load_model(model_path, custom_objects={'total_loss': loss.total_loss})
        model.compile(optimizer=Adam(learning_rate, beta_1=0.5, beta_2=0.995), loss=loss.total_loss, run_eagerly=True)

    # train model
    if test_overfit:
        model.fit(train_dataset, epochs=epoch, callbacks=[reduce_lr_cb, write_images_cb])
        model.save(os.path.join(os.path.dirname(ckpt_path), 'test_overfit'))
    else:
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epoch,
            callbacks=[tensorboard_cb, ckpt_cb, reduce_lr_cb, write_images_cb, early_stop_cb]
        )
        model.save(os.path.join(os.path.dirname(ckpt_path), 'end_train'))

# python train.py --train-annot-dir=dataset/VOC2012_train/Annotations --train-img-dir=dataset/VOC2012_train/JPEGImages --val-annot-dir=dataset/VOC2012_val/Annotations --val-img-dir=dataset/VOC2012_val/JPEGImages --batch-size=16 --epoch=20 --learning-rate=1e-3 --model-path=checkpoints/v1-fast/end_train