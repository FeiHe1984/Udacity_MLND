
import os
import cv2
import numpy as np
import pandas as pd
import csv
import scipy.misc

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sklearn

DATA_PATH = './epochs/'
TRAIN_PATH = './train_images/'
TEST_PATH = './test_images/'
VG_PATH = './vg_images/'
CROP_SHAPE = 300
RESIZED_SHAPE = (80, 80)


def crop_image(image):
    '''
    Crop the sky of the image 
    '''    
    img_shape = image.shape
    img_crop = image[CROP_SHAPE:img_shape[0], 0:img_shape[1]]
    return img_crop

def resized_image(image):
    '''
    Resized image
    '''      
    return cv2.resize(image, RESIZED_SHAPE)

def nomorlize_image(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [0.1, 0.9]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = 0.1
    b = 0.9
    return a + (image_data - 0)*(b - a) / 255.0


def load_data(mode='train', write_to_disk=False, print_shape=False):
    '''
    Load all video data and string data 
    1-9 videos and string csv as train, 10 video and string csv as test 
    and stroe all the images as 'jpg' file and the pathes and 'wheel' as 'csv' file to the disk 
    '''
    print('load data start!')

    if mode == 'train':
        epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if write_to_disk == True:
            if not os.path.exists(TRAIN_PATH):
                os.makedirs(TRAIN_PATH)
            with open(TRAIN_PATH + "train_images.csv","w", newline='') as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerow(["wheel","img_path"])
                csvfile.close()

    elif mode == 'test':
        epochs = [10]
        if write_to_disk == True:
            if not os.path.exists(TEST_PATH):
                os.makedirs(TEST_PATH)
            with open(TEST_PATH + "test_images.csv","w", newline='') as csvfile: 
                writer = csv.writer(csvfile)
                writer.writerow(["wheel","img_path"])
                csvfile.close()
    else:
        print('Wrong mode input')

    imgs = []
    wheels = []
    # extract image and steering data
    for epoch_id in epochs:
        print("The epoch%d mkv is processing" % (epoch_id)) 
        wheel = []

        vid_path = os.path.join(
            DATA_PATH, 'epoch{:0>2}_front.mkv'.format(epoch_id))
        cap = cv2.VideoCapture(vid_path)

        csv_path = os.path.join(
            DATA_PATH, 'epoch{:0>2}_steering.csv'.format(epoch_id))
        rows = pd.read_csv(csv_path)
        wheel = rows['wheel'].values
        wheels.extend(wheel)

        train_images_path = os.path.join(
            TRAIN_PATH, 'epoch{:0>2}_steering_'.format(epoch_id))
        test_images_path = os.path.join(
            TEST_PATH, 'epoch{:0>2}_steering_'.format(epoch_id))

        i = 0
        while True:
            ret, img = cap.read()
            if not ret:
                break
            #Prepocess the image
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = crop_image(img)
            img = resized_image(img)
            #img = nomorlize_image(img)
            imgs.append(img)

            # Write to the disk
            if mode=='train' and write_to_disk==True:
                with open(TRAIN_PATH + "train_images.csv","a", newline='') as csvfile: 
                    writer = csv.writer(csvfile)
                    writer.writerow([wheel[i], train_images_path+str(i)+'.jpg'])
                cv2.imwrite(train_images_path+str(i)+'.jpg', img)
                i = i+1
            if mode=='test' and write_to_disk==True:
                with open(TEST_PATH + "test_images.csv","a", newline='') as csvfile: 
                    writer = csv.writer(csvfile)
                    writer.writerow([wheel[i], test_images_path+str(i)+'.jpg'])
                cv2.imwrite(test_images_path+str(i)+'.jpg', img)
                i = i+1

        assert len(imgs) == len(wheels)
        if print_shape == True:
            print("The epoch%d mkv has %d images" % (epoch_id, len(wheel)))

        cap.release()

    imgs = np.array(imgs)
    wheels = np.array(wheels)
    wheels = np.reshape(wheels,(len(wheels),1))

    print('loading data filished!')

    return imgs, wheels


def write_vg_images(images, samples):
    '''
    Write the images of the vae+gan model created to the disk  
    '''
    # Create path and csv file
    if not os.path.exists(VG_PATH):
        os.makedirs(VG_PATH)
    with open(VG_PATH + "vg_images.csv","w", newline='') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["wheel","img_path"])
        csvfile.close()

    # Write to the disk
    i = 0
    for img in images:
        img = cv2.resize(img, (80, 80))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        with open(VG_PATH + "vg_images.csv","a", newline='') as csvfile: 
            writer = csv.writer(csvfile)
            writer.writerow([samples[i][0], VG_PATH+"vg_images"+str(i)+'.jpg'])
        #cv2.imwrite(VG_PATH+"vg_images"+str(i)+'.jpg', img)
        scipy.misc.imsave(VG_PATH+"vg_images"+str(i)+'.jpg', img)
        i = i+1
    print("Write vg images success!")


def read_csv(csv_path):
    """
    Reading the csv file and store in the list
    """
    samples = []

    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        samples = samples[1:]
        
    return samples

def generator(samples, batch_size=32):
    """
    The generator of the samples
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:                
                name = batch_sample[1]
                image = cv2.imread(name) 
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = nomorlize_image(image)
                images.append(image)   

                measurment = float(batch_sample[0])
                angles.append(measurment)
                
            # Augment and flip the image
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                # flip
                augmented_images.append(cv2.flip(image, 1))
                augmented_angles.append(angle*-1.0)
                # noise
                noisy_img = image + 0.05 * np.random.randn(*image.shape)
                noisy_img = np.clip(noisy_img, 0.1, 0.9)
                augmented_images.append(noisy_img)
                augmented_angles.append(angle)

            # Form the X train and y train and shuffle the generator
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles) 
            yield sklearn.utils.shuffle(X_train, y_train)