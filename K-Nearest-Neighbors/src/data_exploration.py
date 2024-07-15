import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from utils import display_image, load_image


def get_avgrage_width_and_lenght(root_directory, class_names):
    dimentions = []
    for class_name in class_names:
        images_names = os.listdir(root_directory+'/'+class_name)
        for image_name in images_names:
            image_path = root_directory+'/'+class_name+'/'+image_name
            with Image.open(image_path) as image:
                dimentions.append(image.size)
                
    avrage_lenght = sum([d[0] for d in dimentions]) / len(dimentions)   
    avrage_width = sum(d[1] for d in dimentions) / len(dimentions)
    return avrage_lenght, avrage_width

def get_image_intensity_statistics(image):
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    median_intensity = np.median(image)
    return mean_intensity, std_intensity, median_intensity


def print_intensity_statistics(mean_intensity, std_intensity, median_intensity):
    print(f'Mean Intensity: {mean_intensity}')
    print(f'Standard Deviation: {std_intensity}')
    print(f'Median Intensity: {median_intensity}')
    
    
def get_sifit_image(blurred_image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(blurred_image, None)
    sift_image = cv2.drawKeypoints(blurred_image, keypoints, None)
    return sift_image


def get_harris_corners_image(blurred_image_float, image):
    harris_corners = cv2.cornerHarris(blurred_image_float, 2, 3, 0.04)
    harris_corners = cv2.dilate(harris_corners, None)
    corner_image = image.copy()
    corner_image[harris_corners > 0.01 * harris_corners.max()] = 255
    return corner_image

def get_orb_image(blurred_image, image):
    orb = cv2.ORB_create()
    keypoints_orb, descriptors_orb = orb.detectAndCompute(blurred_image, None)
    orb_image = cv2.drawKeypoints(image, keypoints_orb, None)
    return orb_image

def reduce_dimensionality_for_clusterign(flattened_images, number_of_components):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(flattened_images)
    return reduced_data 

def apply_kmeans_clustering(reduced_data, number_of_clusters):
    kmeans = KMeans(n_clusters=number_of_clusters)
    kmeans.fit(reduced_data)
    labels = kmeans.labels_ 
    return labels 
            
            
def image_canny_edge_detection(blurred_image):
    edges = cv2.Canny(blurred_image, 100, 200)
    display_image(edges, 'Canny Edge Detection', 'x', 'y')
    
    
def gaussian_blure():
    image = load_image('../datasets/mnist_all/0/0_1.png')
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Ensure the blurred image is of type CV_8UC1
    blurred_image_float = blurred_image.astype('float32')
    display_image(blurred_image_float, 'Random Image', 'x', 'y')
    return blurred_image, blurred_image_float
    
    
def thresholding(image):
    return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)