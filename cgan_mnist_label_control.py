import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Embedding, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Parameters
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
num_classes = 10
latent_dim = 100

# Create images/ directory if not exists
os.makedirs("images", exist_ok=True)

# Optimizer
optimizer = Adam(0.0002, 0.5)

# ---------------------
#  Build Discriminator
# ---------------------
def build_discriminator():
    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    x = Dense(512)(model_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    validity = Dense(1, activation='sigmoid')(x)

    model = Model([img, label], validity)
    return model

# -----------------
#  Build Generator
# -----------------
def build_generator():
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))
    model_input = multiply([noise, label_embedding])

    x = Dense(256)(model_input)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(np.prod(img_shape), activation='tanh')(x)
    img = Reshape(img_shape)(x)

    model = Model([noise, label], img)
    return model

# Build models
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

generator = build_generator()

# Combined model (Generator + Discriminator)
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([noise, label])
discriminator.trainable = False
validity = discriminator([img, label])

combined = Model([noise, label], validity)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

# ------------
#  Training
# ------------
def train(epochs, batch_size=128, sample_interval=1000):
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)

    half_batch = int(batch_size / 2)

    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------
        idx = np.random.randint(0, X_train.shape[0], half_batch)
        imgs, labels = X_train[idx], y_train[idx]

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        gen_labels = np.random.randint(0, num_classes, half_batch).reshape(-1, 1)
        gen_imgs = generator.predict([noise, gen_labels])

        d_loss_real = discriminator.train_on_batch([imgs, labels.reshape(-1, 1)], np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch([gen_imgs, gen_labels], np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # -----------------
        #  Train Generator
        # -----------------
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
        g_loss = combined.train_on_batch([noise, sampled_labels], np.ones((batch_size, 1)))

        # Print progress
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc.: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

        # Save images
        if epoch % sample_interval == 0:
            save_images(epoch)

# Save grid of generated images
def save_images(epoch):
    r, c = 2, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    labels = np.array([i for i in range(num_classes)]).reshape(-1, 1)

    gen_imgs = generator.predict([noise, labels])
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    count = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[count, :, :, 0], cmap='gray')
            axs[i, j].set_title(f"Digit: {labels[count][0]}")
            axs[i, j].axis('off')
            count += 1
    fig.savefig(f"images/mnist_{epoch}.png")
    plt.close()

# Run training
train(epochs=10000, batch_size=64, sample_interval=1000)
