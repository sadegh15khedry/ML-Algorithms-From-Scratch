from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, test_dataset):
     # Get the true labels
    true_labels = np.concatenate([y.numpy() for x, y in test_dataset], axis=0)
    true_labels = np.argmax(true_labels, axis=1)  # Convert one-hot to class indices
    
    # Make predictions
    predictions = model.predict(test_dataset)
    predicted_labels = np.argmax(predictions, axis=1)
    
    test_loss, test_acc = model.evaluate(test_dataset)
    cm = confusion_matrix(true_labels, predicted_labels)
    
     # Calculate F1 score, precision, and recall
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    
    report = classification_report(true_labels, predicted_labels, target_names=test_dataset.class_names)
    print(report)
    
    # predictions = model.predict(x_test)
    # report = classification_report(y_test, predictions)
    # cm = confusion_matrix(y_test, predictions)
    return test_loss, test_acc, cm, true_labels, predicted_labels, f1, precision, recall


def get_test_dataset(test_dir, image_width, image_height):
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=(image_height, image_width),
    color_mode='grayscale',
    label_mode='categorical'  # or 'categorical' for one-hot encoded labels
    )
    return test_dataset

def display_and_save_confution_matrix(cm, test_dataset, file_path):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.class_names, yticklabels=test_dataset.class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.show()

