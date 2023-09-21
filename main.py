# Gerekli kütüphaneleri içe aktarın
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Veri yolu ve eğitim/test bölünmesi
data_path = "data_set"  # Kedi ve köpek görüntülerinin bulunduğu klasör
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Görüntü piksellerini normalize etmek için 0-1 aralığına ölçeklendirme
    rotation_range=40,  # Rastgele döndürme
    width_shift_range=0.2,  # Genişlik kaydırma
    height_shift_range=0.2,  # Yükseklik kaydırma
    shear_range=0.2,  # Kesme dönüşümü
    zoom_range=0.2,  # Rastgele yakınlaştırma
    horizontal_flip=True,  # Yatay çevirme
    fill_mode='nearest'  # Piksel doldurma yöntemi
)

# Veri yolu ve eğitim/test bölünmesi
train_generator = train_datagen.flow_from_directory(
    data_path + '/training_set',
    target_size=(150, 150),  # Görüntü boyutlarını yeniden boyutlandırma
    batch_size=32,
    class_mode='binary'  # İkili sınıflandırma
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    data_path + '/training_set',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Modeli oluşturma
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Modeli derleme
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Eğitim
history = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50
)

# Modelin değerlendirilmesi
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    data_path + '/test_set',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)


test_loss, test_accuracy = model.evaluate(test_generator, steps=50)
#print("Test accuracy:", test_accuracy)

# Tahmin
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

test_image = image.load_img('cat.jpg', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)

if result[0][0] >= 0.5:
    prediction = 'Köpek (Dog) - Olasılık: {:.2f}%'.format(result[0][0] * 100)
else:
    prediction = 'Kedi (Cat) - Olasılık: {:.2f}%'.format((1 - result[0][0]) * 100)

print(prediction)
