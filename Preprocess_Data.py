import os
import wget
from zipfile import PyZipFile
import shutil
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array
import cv2
import h5py
import glob

temp_directory = 'data/temp'
train_directory = 'data/train'
aug_images_directory = train_directory + '/augumented_images'
# Url to download DrivFace dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00378/DrivFace.zip'
# File name of the output of pre processing
hdf5_path = 'DrivFace.h5'
# Shape of input data
HEIGHT = 224
WIDTH = 224
CHANNELS = 3
SHAPE = (HEIGHT, WIDTH, CHANNELS)


def load_dataset():
  if not os.path.exists(train_directory):
    os.makedirs(temp_directory)
    os.makedirs(train_directory)
    print("Downloading dataset to "+ temp_directory)
    file = wget.download(url, out=temp_directory)
    print("\nUnzipping the files..")
    pzf = PyZipFile(file)
    pzf.extractall(temp_directory)
    pzf = PyZipFile(temp_directory+'/DrivFace/DrivImages.zip')
    pzf.extractall(temp_directory)
    print("Moving files to "+train_directory)
    for file in os.listdir(temp_directory+'/DrivImages'):
      shutil.move(temp_directory+'/DrivImages/'+file,train_directory+'/'+file)
    shutil.move(temp_directory+'/DrivFace/drivPoints.txt',train_directory+'/drivPoints.txt')
    print("Deleting temporary directory "+ temp_directory)
    shutil.rmtree(temp_directory)

def create_augumented_images(directory):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        return
    print("Generating augumented images in the directory " + directory)
    for index, row in df_label.iterrows():
            file = row['fileName'] + ".jpg"
            image = cv2.imread(train_directory + '/' + file)
            y = int(row['yF'])
            x = int(row['xF'])
            w = int(row['wF'])
            h = int(row['hF'])
            #image = image[y:y+h, x:x+w]
            image = cv2.resize(image, (HEIGHT, WIDTH), interpolation = cv2.INTER_AREA)
            X = img_to_array(image)
            X = X.reshape((1,) + X.shape)
            i = 0
            for batch in datagen.flow(X, batch_size=1,
                              save_to_dir=directory, save_prefix=file, save_format='jpg'):
                i += 1
                if i > 20:
                    break  # otherwise the generator would loop indefinitely

def proc_images():
    num_images = 0
    dataset_X = []
    dataset_y = []
    if os.path.isfile(hdf5_path):
        return
    else:
        print("writing images to " + hdf5_path)
        with h5py.File(hdf5_path, mode='w') as hf:
            for index, row in df_label.iterrows():
                file = row['fileName'] + ".jpg"
                image = cv2.imread(train_directory + '/' + file)
                y = int(row['yF'])
                x = int(row['xF'])
                w = int(row['wF'])
                h = int(row['hF'])
                #image = image[y:y+h, x:x+w]
                image = cv2.resize(image, (HEIGHT, WIDTH), interpolation = cv2.INTER_AREA)
                dataset_X.append(image)
                dataset_y.append(row['label'])
                num_images = num_images + 1
                for augumented_image in glob.glob(train_directory + '/augumented_images/' + file + "*"):
                    image = cv2.imread(augumented_image)
                    y = int(row['yF'])
                    x = int(row['xF'])
                    w = int(row['wF'])
                    h = int(row['hF'])
                    #image = image[y:y+h, x:x+w]
                    image = cv2.resize(image, (HEIGHT, WIDTH), interpolation = cv2.INTER_AREA)
                    dataset_X.append(image)
                    dataset_y.append(row['label'])
                    num_images = num_images + 1
            dataset_X = hf.create_dataset(
                    name='dataset_X',
                    data=dataset_X,
                    shape=(len(dataset_X), HEIGHT, WIDTH, CHANNELS),
                    maxshape=(len(dataset_X), HEIGHT, WIDTH, CHANNELS),
                    compression="gzip",
                    compression_opts=9)
            dataset_y = hf.create_dataset(
                    name='dataset_y',
                    data=dataset_y,
                    shape=(len(dataset_y), 1,),
                    maxshape=(len(dataset_y), None,),
                    compression="gzip",
                    compression_opts=9)
            number_images = next(os.walk(train_directory))[2]
            number_images_aug = next(os.walk(aug_images_directory))[2]
            print("Number of images written to "+ hdf5_path + " is: " + str(len(number_images)+len(number_images_aug)))

load_dataset()
df_label = pd.read_csv(train_directory+'/drivPoints.txt')
df_label = df_label.dropna()
create_augumented_images(aug_images_directory)
proc_images()
