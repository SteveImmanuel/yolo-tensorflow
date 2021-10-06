from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Input
from yolo.v1.config import LEARNING_RATE
from yolo.v1.model import FastYoloV1Model
from yolo.v1.losses import YoloV1Loss
from yolo.pascal_voc_dataset import PascalVOCDataset

if __name__ == '__main__':

    model = FastYoloV1Model()
    loss = YoloV1Loss()
    model.build(input_shape=(None, 448, 448, 3))
    model.call(Input(shape=(448, 448, 3)))
    model.summary()
    model.compile(optimizer=Adam(LEARNING_RATE), loss=loss.total_loss, run_eagerly=True)

    ckpt_cb = ModelCheckpoint('checkpoints/v1', monitor='val_loss', mode='min')
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=3, verbose=1, mode='min')

    train_dataset = PascalVOCDataset('dataset/VOC2012_train/JPEGImages', 'dataset/VOC2012_train/Annotations', batch_size=16)
    val_dataset = PascalVOCDataset('dataset/VOC2012_val/JPEGImages', 'dataset/VOC2012_val/Annotations', batch_size=16)

    model.fit(train_dataset, validation_data=val_dataset, epochs=20, callbacks=[ckpt_cb, reduce_lr_cb])