from sklearn.linear_model import LogisticRegression

from logestic_regression import CustomLogesticRegression


def train_model(x_train, y_train, number_of_iterations, learning_rate, model_type='custom'):
    if model_type == 'custom':
        model = CustomLogesticRegression(learning_rate, number_of_iterations)
        model.fit(x_train, y_train)
        return model
    
    elif model_type == 'sklearn':
        model = LogisticRegression ()
        model.fit(x_train, y_train)
        return model


def get_train_dataset(train_dir, batch_size, image_width, image_height, augmentation):
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(image_height, image_width),
    batch_size=batch_size,
    color_mode='grayscale',
    label_mode='categorical'  # or 'categorical' for one-hot encoded labels
    )
    if augmentation:
        train_dataset = train_dataset.map(, num_parallel_calls=tf.data.AUTOTUNE)
    return train_dataset



