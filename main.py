from model import unet
from data import m2nist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

width, height = 80, 64
(train_imgs, train_masks), (val_imgs, val_masks) = m2nist('data/m2nist2', (width, height), 0.2)
model = unet(input_size=(height, width, 1))
opt = Adam(learning_rate=0.001, decay=1e-5)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
csv_logger = CSVLogger('trainlog.csv')
print(train_imgs.shape, train_masks.shape)
print(val_imgs.shape, val_masks.shape)
model.fit(
    train_imgs,
    train_masks,
    # validation_data=(val_imgs, val_masks),
    # validation_steps=200,
    steps_per_epoch=1000,
    epochs=50,
    callbacks=[csv_logger])
