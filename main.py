import numpy as np
import requests

from PIL import Image
from io import BytesIO

from tensorflow.keras.applications.vgg16 import (
    VGG16,
    preprocess_input,
    decode_predictions
)
IMAGE_URL = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800"
def download_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

print("Caricamento modello VGG16...")

model = VGG16(
    weights="imagenet",
    include_top=True
)

print("Modello caricato correttamente!")
img = download_image(IMAGE_URL)
def prepare_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    img_array = preprocess_input(img_array)

    return img_array

print("Immagine caricata!")
print(img.size)
prepared_img = prepare_image(img)

print("Immagine preprocessata!")
print(prepared_img.shape)
predictions = model.predict(prepared_img)

results = decode_predictions(predictions, top=5)[0]

print("\n" + "=" * 50)
print("TOP 5 PREDIZIONI VGG16")
print("=" * 50)

for index, (_, label, probability) in enumerate(results, start=1):
    print(f"{index}. {label:<20} {probability * 100:.2f}%")