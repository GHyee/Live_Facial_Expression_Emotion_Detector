from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from tensorflow.python.client import device_lib
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
parent_path = os.path.dirname(__file__)
src_path = os.path.dirname(parent_path)
sys.path.append(parent_path)
sys.path.append(src_path)

class CustomModel:
    def __init__(self, train_dir, val_dir, test_dir):

        self.batchsize = 32
        self.img_height = 64  # 224
        self.img_width = 64  # 224
        self.channels = 3

        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir


    def import_augmented_data(self):
        """data augmentation on train/val/test sets
        """
        self.train_datagen = ImageDataGenerator(rotation_range=20.0,
                                                zoom_range=0.15,
                                                horizontal_flip=True,
                                                width_shift_range=0.2,
                                                height_shift_range=0.2,
                                                shear_range=0.15)


        self.test_datagen = ImageDataGenerator()

        # train set
        self.train_generator = self.train_datagen.flow_from_directory(self.train_dir,
                                                            target_size=(self.img_height, self.img_width),
                                                            batch_size= self.batchsize,
                                                            class_mode='categorical',
                                                            shuffle=True)

        # validation set
        self.validation_generator = self.test_datagen.flow_from_directory(self.val_dir,
                                                                target_size=(self.img_height, self.img_width),
                                                                batch_size=self.batchsize,
                                                                class_mode='categorical')

        # test set
        self.test_generator = self.test_datagen.flow_from_directory(self.test_dir,
                                                                target_size=(self.img_height, self.img_width),
                                                                batch_size=self.batchsize,
                                                                class_mode='categorical')


        return self


    def create_model(self):
        """
        use MobileNetV2 as a base model and fine tune the top layers
        """

        baseModel = MobileNetV3Small(weights="imagenet", include_top=False,
                                     input_tensor=Input(shape=(64, 64, 3)))  #  input_tensor=Input(shape=(224, 224, 3)))

        # construct the head of the model that will be placed on top of the
        # the base model
        headModel = baseModel.output
        #headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = BatchNormalization()(headModel)
        headModel = Dense(128, activation="relu")(headModel)
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(3, activation="softmax")(headModel)

        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        model = Model(inputs=baseModel.input, outputs=headModel)

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the first training process
        for layer in baseModel.layers:
            layer.trainable = False


        INIT_LR = 1e-4
        EPOCHS = 10

        # compile our model
        print("[INFO] compiling model...")
        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        self.model = model
        return self.model.summary()


    def train_model(self, model_output_path, epochs=1):
        """
        commence model training

        """

        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            verbose=1,
            patience=3,
            min_delta=0.00001
        )

        checkpoint = ModelCheckpoint(
            filepath=model_output_path,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
            )

        self.model.fit(self.train_generator,
                        steps_per_epoch = self.train_generator.samples // self.batchsize,
                        validation_data = self.validation_generator,
                        validation_steps = self.validation_generator.samples // self.batchsize,
                        epochs = epochs,
                        callbacks=[early_stopping, checkpoint])

        return self


    def predict(self):
        """
        compute test accuracy
        """

        test_loss, test_accuracy = self.model.evaluate(self.test_generator)

        return test_loss, test_accuracy



    # def save_model(self):
        """polyaxon"""


if __name__ == '__main__':

    # 1) change directory into the team5 directory
    if os.path.split(os.getcwd())[1] == 'modelling':
        os.chdir('..')
    if os.path.split(os.getcwd())[1] == 'src':
        os.chdir('..')
    logger.info(os.getcwd())

    logger.info(device_lib.list_local_devices())

    # 2) define the directories
    train_dir = os.path.join('images', 'train')
    val_dir = os.path.join('images', 'valid')
    test_dir = os.path.join('images', 'test')

    #3) define the model
    model = CustomModel(train_dir, val_dir, test_dir)
    model.import_augmented_data()
    model.create_model()

    #4) train model
    model.train_model('faces_30.h5', epochs=30)

    # 5) test model
    test_loss, test_accuracy = model.predict()

    logger.info(test_loss)
    logger.info(test_accuracy)
