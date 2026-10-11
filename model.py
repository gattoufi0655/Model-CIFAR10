# ==========================================
# IMPORT DES LIBRARIES
# ==========================================
import tensorflow as tf
from keras import datasets, layers, models
from keras.applications import EfficientNetB0
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
#   CHARGEMENT DU DATASET CIFAR-10
# ==========================================
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Classes
class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

print("Dataset chargé avec succès")

# ==========================================
#   OPTIMISATION AVEC tf.data 
# ==========================================
BATCH_SIZE = 64

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# ==========================================
#  CRÉATION DU MODÈLE 
# ==========================================
def create_model():

    # Modèle pré-entraîné EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(160, 160, 3)  #  taille réduite pour accélérer
    )

    #  Geler les couches (Transfer Learning)
    base_model.trainable = False

    #  Modèle complet
    model = models.Sequential([

        # Normalisation + Resize
        layers.Rescaling(1./255),
        layers.Resizing(160, 160),

        #  Data augmentation (légère pour vitesse)
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),

        # Extraction des features
        base_model, # EfficientNetB0 pré-entraîné sur ImageNet

        # Réduction des dimensions
        layers.GlobalAveragePooling2D(),

        #  Classification 
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        # Sortie (10 classes)
        layers.Dense(10, activation='softmax')
    ])

    return model


# Création du modèle
model = create_model()

# Affichage architecture
print("\n=== ARCHITECTURE DU MODÈLE ===")
model.summary()

# ==========================================
#  COMPILATION DU MODÈLE
# ==========================================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
#  CALLBACKS (OPTIMISATION)
# ==========================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

# ==========================================
#  ENTRAÎNEMENT
# ==========================================
print("\n=== ENTRAÎNEMENT ===")

history = model.fit(
    train_ds, # entraînement optimisé avec tf.data et data augmentation intégrée
    epochs=8,  # rapide
    validation_data=test_ds,
    callbacks=[early_stop, reduce_lr], # callbacks pour optimiser l'entraînement et éviter le surapprentissage
    verbose=1 # affichage détaillé pour suivre l'entraînement
)

# ==========================================
#  ÉVALUATION
# ==========================================
test_loss, test_acc = model.evaluate(test_ds)

print("\n=== RÉSULTATS ===")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# ==========================================
#  AFFICHAGE DES COURBES
# ==========================================
plt.figure(figsize=(12,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend()

plt.show()

# ==========================================
#  PRÉDICTION RAPIDE
# ==========================================
def fast_ensemble(model, images):
    preds1 = model.predict(images, verbose=0)
    preds2 = model.predict(tf.image.flip_left_right(images), verbose=0)
    return (preds1 + preds2) / 2

# Test sur 100 images
sample_images = test_images[:100]
ensemble_pred = fast_ensemble(model, sample_images)

confidence = np.max(ensemble_pred, axis=1)

print("\n=== CONFIANCE ===")
print(f"Moyenne: {np.mean(confidence)*100:.2f}%")

# ==========================================
#  PRÉDICTION VISUELLE
# ==========================================
predictions = model.predict(test_images[:10])

plt.figure(figsize=(12,6))

for i in range(10):
    plt.subplot(2,5,i+1)
    
    pred_class = np.argmax(predictions[i])
    true_class = test_labels[i][0]
    
    plt.imshow(test_images[i])
    plt.title(f"{class_names[pred_class]}")
    plt.axis('off')

plt.show()

# ==========================================
#  SAUVEGARDE DU MODÈLE
# ==========================================
model.save("cifar_fast_model.keras")
print("\n✅ Modèle sauvegardé : cifar_fast_model.keras")

# ==========================================
#  FIN
# ==========================================
print("\n Modèle prêt pour test réel et soutenance !")