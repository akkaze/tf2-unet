from model import unet
from data import m2nist
import os
from losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

width, height = 80, 64
train_ds, val_ds = m2nist('data/m2nist', (width, height), 0.2)
model = unet(input_size=(height, width, 1), num_classes=11)
opt = Adam(learning_rate=0.001, decay=1e-5)
os.makedirs('checkpoints', exist_ok=True)
callbacks = [
    CSVLogger('trainlog.csv'),
    EarlyStopping(patience=5, verbose=1),
    ModelCheckpoint('checkpoints/unet_train_{epoch}.h5', verbose=1, save_weights_only=True),
    TensorBoard(log_dir='logs')
]
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
history = model.fit(train_ds,
                    epochs=20,
                    steps_per_epoch=200,
                    callbacks=callbacks,
                    validation_data=val_ds,
                    validation_steps=20)
