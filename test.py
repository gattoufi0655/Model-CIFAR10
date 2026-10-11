import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf

# Charger modèle
model = tf.keras.models.load_model("cifar_fast_model.keras")

class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

folder_path = "images"

images_list = []
titles_list = []
colors_list = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):

        img_path = os.path.join(folder_path, filename)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #  taille modèle
        image_resized = cv2.resize(image, (160, 160))

        image_normalized = image_resized.astype("float32") / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)

        prediction = model.predict(image_input, verbose=0)

        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        pred_name = class_names[predicted_class]

        # Vérification simple (si nom fichier contient classe)
        if pred_name in filename.lower():
            color = "green"
        else:
            color = "red"

        # Texte affiché
        title = f"{pred_name} ({confidence*100:.1f}%)\n[{filename}]"

        images_list.append(image_resized)
        titles_list.append(title)
        colors_list.append(color)

# ===============================
# AFFICHAGE GRID PROPRE
# ===============================

n_images = len(images_list)
cols = 5
rows = (n_images // cols) + 1

plt.figure(figsize=(15, 3 * rows))

for i in range(n_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images_list[i])
    plt.title(titles_list[i], fontsize=9, color=colors_list[i])
    plt.axis('off')

plt.suptitle("Résultats du modèle sur images réelles", fontsize=16)
plt.tight_layout()
plt.show()