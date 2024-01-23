import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

class Schmoo:
  def __init__(self) -> None:
    self.training_data_path = './train'
    self.validation_data_path = './val'
    self.model_path = 'schmoo.h5'

  def U_Net_Model(self, input_size=(256, 256, 3)):
    inputs = Input(input_size)
    
    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Decoder
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    up1 = UpSampling2D(size=(2, 2))(conv2)
    
    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    self.model = model

  def Train_Model(self):
    Schmoo.U_Net_Model

    # Create data generators
    train_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        self.train_data_path,
        target_size=(256, 256),
        batch_size=32,
        class_mode='input',  # assuming your masks are the same size as images
    )

    val_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
        self.validation_data_path,
        target_size=(256, 256),
        batch_size=32,
        class_mode='input'
    )

    self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    self.model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )

    self.model.save(self.model_path)

  def Test_Model(self, imagePath: str = './val/image.jpg'):
    model = load_model(self.model_path)

    img = load_img(imagePath, target_size=(256, 256))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) 
    img_array /= 255.0  # Normalize pixel values

    # Perform inference
    predictions = model.predict(img_array)

    # Display the original image and segmentation mask
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(predictions[0, :, :, 0], cmap='gray')
    plt.title('Segmentation Mask')

    plt.show()

if __name__ == "__main__":
  Schmoo().Test_Model()