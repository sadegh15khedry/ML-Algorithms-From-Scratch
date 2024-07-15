import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import tensorflow as tf


 
def save_report(report, file_path):
    with open(file_path, 'w') as f:
        f.write(report)
        
        
def save_dataframe_as_csv(df, file_path):
    df.to_csv(file_path, index=False)


def save_model(model, path):
    model.save(path)

def load_saved_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    

def load_image(image_directory):
    image = cv2.imread(image_directory, cv2.IMREAD_GRAYSCALE)
    # image = cv2.imread(image_directory)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #converting BGR to RGB
    return image

def remove_image(image_path):
    os.remove(image_path)
    
    
def get_image_histogram(image):
    # Calculate the image histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Normalize the image histogram 
    hist /= hist.sum()
    return hist

def display_image(image, title, xlabel, ylabel, cmap='gray'):
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    xlabel= xlabel
    ylabel= ylabel
    plt.axis('off')
    plt.show()
    
def display_histogram(hist, title, color, xlabel, ylabel):
    plt.plot(hist, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def display_scatter(x_data, y_data, labels, title, xlabel, ylabel):
    plt.scatter(x_data,y_data, c=labels, cmap='viridis')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def flatten_images(images):
    flattened_images = []
    for imgage in images:
        flattened_images.append(imgage.flatten())
    return np.array(flattened_images)


def load_all_images(source_directory, class_names):
    # Loading all the images
    images = []
    for class_name in class_names:
        image_names = os.listdir(source_directory+class_name)
        for image_name in image_names:
            image = cv2.imread(source_directory+class_name+'/'+image_name)
            images.append(image)
            
    return images



    # Function to save images
def save_images(data, labels, folder):
    os.makedirs(folder, exist_ok=True)
    for i, (image, label) in enumerate(zip(data, labels)):
        path = folder + '/' + str(label)+ '/' + str(label) + '_'+ str(i)+'.png'
        img = Image.fromarray(image)
        img.save(path)

     
def download_splited_minst_dataset():        
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Concatenate train and test data
    x_all = np.concatenate((x_train, x_test))
    y_all = np.concatenate((y_train, y_test))

    # Save all images
    save_images(x_all, y_all, "../datasets/mnist_all")

    print("MNIST images saved successfully.")
    
def check_tensorflow():
    #making sure the gpu is available
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))