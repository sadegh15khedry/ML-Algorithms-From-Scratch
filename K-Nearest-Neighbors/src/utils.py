import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import cv2
import os
import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from sklearn.preprocessing import MultiLabelBinarizer
 
def save_report(report, file_path):
    with open(file_path, 'w') as f:
        f.write(report)
        
        
def save_dataframe_as_csv(df, file_path):
    df.to_csv(file_path, index=False)


def save_model(model, path):
    model.save(path)


    

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
 
    
    
    
def get_error(y_train, y_pred_train):
    return mean_squared_error(y_train, y_pred_train)   

def get_accuracy(y_test, y_pred):
    
    # Convert y_test to binary format
    y_test_binary, mlb = convert_to_binary_format(y_test)
    # Convert y_pred to binary format using the same MultiLabelBinarizer
    y_pred_binary, _ = convert_to_binary_format(y_pred, mlb)
    
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    print(f"Accuracy: {accuracy:.2f}")
    
    precision = precision_score(y_test_binary, y_pred_binary, average='weighted')
    print(f"Precision: {precision:.2f}")
    
    recall = recall_score(y_test_binary, y_pred_binary, average='weighted')
    print(f"Recall: {recall:.2f}")
    
    f1 = f1_score(y_test_binary, y_pred_binary, average='weighted')
    print(f"F1 Score: {f1:.2f}")
    
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.2f}")
    # precision = precision_score(y_test, y_pred, average='weighted')
    # print(f"Precision: {precision:.2f}")
    # recall = recall_score(y_test, y_pred, average='weighted')
    # print(f"Recall: {recall:.2f}")
    # f1 = f1_score(y_test, y_pred, average='weighted')
    # print(f"F1 Score: {f1:.2f}")
    
    
    
def get_image_dataset(dir, image_width, image_height, labels):
    # Initialize lists to hold image data and labels
    images = []
    images_label = []
    
    # Loop over each directory (which represents a label)
    for label in labels:
        label_dir = dir + label
        for file_name in os.listdir(label_dir):
            # Load the image
            image_path = os.path.join(label_dir, file_name)
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Resize to 28x28 (if necessary)
            image = image.resize((image_width, image_height))
            
            # Convert the image to a numpy array and flatten it
            image_array = np.array(image).flatten()
            
            # Append the image array and label to lists
            images.append(image_array)
            images_label.append(label)

    # Convert lists to numpy arrays
    print(len(images_label))

    x = np.array(images)
    y = np.array(images_label)

    # Split the data into training and test sets
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return x, y


def convert_to_binary_format(y, mlb=None):
    if mlb is None:
        mlb = MultiLabelBinarizer()
        y_binary = mlb.fit_transform(y)
    else:
        y_binary = mlb.transform(y)
    return y_binary, mlb