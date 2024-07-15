import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt


def get_untrained_custom_model(imgage_width, imgage_height, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
    model = Sequential()
    # Convolutional layer
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(imgage_width, imgage_height, 1), kernel_regularizer=l2(0.001)))  
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(imgage_width, imgage_height, 1), kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2, 2)))   
    model.add(Flatten())
    # Dense layers
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(rate=0.5))
    # Output layer
    model.add(Dense(units=10, activation='softmax'))
    
    
    model.compile(optimizer=optimizer, loss=loss , metrics=metrics)
    return model



def train_model(model, train_dataset, epochs, val_dataset, augmentation=False):
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )
    return history



def augment_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    # Skiping saturation and hue adjustments for grayscale images
    # image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    # image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image = tf.image.resize(image, [224, 224])  # Ensure the size is correct
    return image, label



def get_train_dataset(train_dir, batch_size, image_width, image_height, augmentation):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='categorical'  # or 'categorical' for one-hot encoded labels
    )
    if augmentation:
        train_dataset = train_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    return train_dataset

def get_val_dataset(val_dir, batch_size, image_width, image_height, augmentation):
    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    color_mode='grayscale',
    image_size=(image_height, image_width),
    batch_size=batch_size,
    label_mode='categorical'  # or 'categorical' for one-hot encoded labels
    )
    if augmentation:
        val_dataset = val_dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    return val_dataset


def plot_training_history(history):
    # Extract data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    # Plot training and validation accuracy
    plt.figure(figsize=(12, 6))
    
    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    # Saving the plots
    plt.savefig('../results/training_validation_loss_and_accuracy.png')
    
    # Display the plot
    plt.show()