import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np

# Załadowanie modelu VGG Net
model = VGG16(weights='imagenet')

# Wczytanie obrazu z pliku
img_path = '/Users/filipczop/Pracainz/Duzy zbior/gory_1.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Dokonanie klasyfikacji obrazu
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=3)[0])