from sklearn.model_selection import train_test_split
import os
import random
import shutil
import cv2
import sys
from PIL import Image
sys.path.append(os.path.abspath('../src'))
from utils import remove_image
       


def calculate_train_val_test_sizes(total_sizes_each_class, val_percetage, test_percetage):
    val_size = [int(x * val_percetage) for x in  total_sizes_each_class]
    test_size = [int(x * test_percetage) for x in  total_sizes_each_class]
    train_size = [(x-y)- z for x, y, z in zip(total_sizes_each_class, val_size, test_size)]
    return train_size, val_size, test_size


        
def split_images_into_train_validation_test(raw_dirctory, target_directory, class_names, val_sizes, test_sizes):
    for class_name in class_names:
        src = raw_dirctory + str(class_name)+'/'
        my_pics = os.listdir(path=src)
        class_index =  class_names.index(class_name)
        for i in range(test_sizes[class_index]):
            pic_name = my_pics.pop(random.randrange(len(my_pics)))
            shutil.copyfile(src=src+pic_name, dst=target_directory+'test/'+class_name+'/' + str(pic_name))
        for i in range(val_sizes[class_index]):
            pic_name = my_pics.pop(random.randrange(len(my_pics)))
            shutil.copyfile(src=src+pic_name, dst=target_directory + 'val/'+class_name+'/' + str(pic_name))
        for i in my_pics:
            pic_name = i
            shutil.copyfile(src=src+pic_name, dst=target_directory + 'train/'+class_name+'/' + str(pic_name))
            

    
def is_image_corrupt(image_path):
    #Checking with Pillow
    try:
        img = Image.open(image_path)
        img.verify()
    except (IOError, SyntaxError) as e:
        print(f"Corrupt image detected with Pillow: {image_path} - {e}")
        return True
    
    #Checking with cv2
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image is corrupt")
    except Exception as e:
        print(f"Corrupt image detected with OpenCV: {image_path} - {e}")
        return True



def remove_corrupt_images(root_directory, class_names):
    removed_list = []
    for class_name in class_names:
        images_names = os.listdir(root_directory+'/'+class_name)
        for image_name in images_names:
            is_corrupt = is_image_corrupt(root_directory+'/'+class_name+'/'+image_name)
            if is_corrupt == True:
                print(image_name+' is corrupted and is gettig removed')
                remove_image(root_directory+'/'+class_name+'/'+image_name)
                removed_list.append(image_name)
                
    return removed_list

def convert_images_to_grayscale (source_root_directory, destination_root_directory, class_names):
    for class_name in class_names:
        image_names = os.listdir(source_root_directory+class_name)
        for image_name in image_names:
            image_source_path = source_root_directory+class_name+'/'+image_name
            image_destination_path = destination_root_directory+class_name+'/'+image_name
            print(image_source_path)
            image = cv2.imread(image_source_path)
            #converitg to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(image_destination_path, gray_image)
            
            