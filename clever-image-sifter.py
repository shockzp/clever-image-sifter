#! /usr/bin/env python
'''
Author : Shakul Ramkumar

The purpose of this program is to search for a target image in a disk image smartly, which will combat
anti-forensic technique such as changing one pixel of an image to change the hash of the whole image.

This program recovers all the files on the disk and finds the n-most similar images to the target image.
For the file recovery, sleuthkit is used at the backend. Then, all the files recovered are sifted for the extension type
and all the target images are obtained. For each image obtained, feature vectors are constructed using
pre-trained models from tensorflow hub library. Using these features, Annoy library is used to apply
nearest neighbour algorithm on these images and n-most similar images to the target image is found. Then these found
images are copied to a folder where the user supplied so that they can view it easily.
'''

#Imports all dependant libraries
import argparse
import os
import glob
import pytsk3 as tsk
import shutil
from termcolor import colored
import tensorflow as tf
import tensorflow_hub as hub
from annoy import AnnoyIndex
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Global Strings
original_recovery_dir = "Original recovery/"


def posix_escape(input_string):
    '''
    This function helps in escaping strings in posix style fashion.
    :param input_string: Ideally, a system path.
    :return: Posix style escaped string.
    '''

    return input_string.replace(" ","\\ ")

def get_all_target_files(directory,format):
    '''
    Finds all the files of a particular format within a directory.
    :param directory: Absolute path of the directory in which the file should be searched
    :param format: The file format to be searched for.
    :return: A list containing absolute path of all the files of a particular format found in the directory.
    '''
    return glob.glob(directory + original_recovery_dir + '**/*.' + format, recursive=True)

def copy_files_to_dir(directory,files):
    '''
    This function copies a list of supplied files to the given target directory.
    :param directory: Target directory into which the files have to be stored.
    :param files: List containing absolute path of files.
    '''
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    for file in files:
        shutil.copy2(file,directory,follow_symlinks=True)

def recover_files(disk_image_path,directory,recovery_type):
    '''
    This function recovers all the files in the given disk image and places the output in the given directory.
    It also takes care of recovery type requested from the user. It uses sluethkit in the backend to perform
    these tasks.
    :param disk_image_path: Absolute path of disk image
    :param directory: Absolute path of the directory where the recovered files has to be stored.
    :param recovery_type: [1]. Recover resident files only [2]. Recover resident and deleted files
    :return: True on success, false otherwise
    '''

    print("Starting recovery of files...")

    #Equivalent to MMLS operation
    img = tsk.Img_Info(url=disk_image_path)
    volume = tsk.Volume_Info(img)

    #Forming a list of file system offset addresses to recover, from the raw disk image.
    supported_file_systems = ['NTFS', 'FAT']
    file_systems_to_recover = []
    for part in volume:
        if any(part.desc.decode('utf-8')[0:3] in fs for fs in supported_file_systems):
            file_systems_to_recover.append(part.start)

    #Recovering the files
    recovery_flag = "e"
    space_character = " "

    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

    if recovery_type == 1:
        recovery_flag = "a"

    for start_addr in file_systems_to_recover:
        command = "tsk_recover" + space_character + "-" + recovery_flag + "o" + space_character + str(start_addr) + \
                  space_character + disk_image_path + space_character + directory + posix_escape(original_recovery_dir)
        ret = os.system(command)
        if ret != 0:
            print("FATAL ERROR in recovery occured.")
            return False

    print("All files have been recovered! You can find them at {}".format(directory +
                                                                          posix_escape(original_recovery_dir)))

    return True

def load_img(image_path,image_format):
    '''
    This function preprocesses the image based on the image format.
    :param image_path: Absolute path of the image.
    :param image_format: Format type of the image. Either jpg or png.
    :return: Preprocessed image.
    '''

    img = tf.io.read_file(image_path)
    if(image_format=='jpg'):
        img = tf.io.decode_jpeg(img, channels=3)
    else:
        img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize_with_pad(img, 224, 224)
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    return img

def make_feature_vectors(all_target_files,image_format):
    '''
    This function generates a feature vector for each image supplied to it.
    :param all_target_files: List containing absolute file paths
    :param image_format: Format of the images given
    :return: featurefile_dict is a dictionary containing feature vectors corresponding to each file provided
                    in all_target_files.
               manual_inspection is a list containing absolute file paths which had errors during processing.
    '''
    #Load a pretrained model
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    module = hub.load(module_handle)

    #Filename, feature vector
    featurefile_dict = {}
    manual_inspection = []

    for filename in all_target_files:
        try:
            # Preprocess the image
            img = load_img(filename,image_format)

            # Calculate the feature vectors
            features = module(img)
            feature_set = np.squeeze(features)
            featurefile_dict[filename] = feature_set
        except:
            manual_inspection.append(filename)
            pass

    return featurefile_dict, manual_inspection

def find_nearerst_neighbour(target_image,n_nearest_neighbors,feature_dict):
    '''
    This function is responsible for finding n-most similar looking images to the target image.
    :param target_image: The image for which nearest neighbours have to be found.
    :param n_nearest_neighbors: parameter controlling how many nearest neighbours have to be found.
    :param feature_dict: dictionary containing file path with their corresponding feature vectors.
    :return: n-nearest neighbours to the target image supplied.
    '''
    #Parameters that were found to work the best for this use case.
    dims = 1792
    trees = 10000
    t = AnnoyIndex(dims, metric='angular')

    file_index_to_file_name = {}
    file_name_to_file_index = {}

    # Reads all file names which stores feature vectors
    all_features_files = feature_dict.keys()

    for file_index, file_name in enumerate(all_features_files):
        # Retrieve feature vectors
        file_vector = feature_dict[file_name]

        file_index_to_file_name[file_index] = file_name
        file_name_to_file_index[file_name] = file_index

        # Adds image feature vectors into annoy index
        t.add_item(file_index, file_vector)


    # Building annoy index
    t.build(trees)

    named_nearest_neighbors = []
    find_index = file_name_to_file_index[target_image]

    #Becuase the target image is the nearest neghbour of itself too, we should ask for one extra neighbour.
    effective_nn_to_find = n_nearest_neighbors+1
    nearest_neighbors = t.get_nns_by_item(find_index, effective_nn_to_find)

    for j in nearest_neighbors:
        named_nearest_neighbors.append(file_index_to_file_name[j])

    #To account for itself in nearest neighbour calculations.
    named_nearest_neighbors.remove(target_image)
    return named_nearest_neighbors


def driver(args):
    '''
    This function is responsible for driving the whole control of the program.
    :param args: parsed arguments to the program when it is run.
    '''

    ret = recover_files(args.input_file,args.output_directory,args.recovery_type)

    if(ret):

        output_dir = posix_escape(args.output_directory)
        all_target_files = get_all_target_files(output_dir, args.image_format)


        target_image = os.path.abspath(posix_escape(args.image_to_find))
        #Adding target image to the list of files so that nearest neighbour of it can be calculated.
        all_target_files.append(target_image)
        print("Started processing files...")
        f_dict,m_list = make_feature_vectors(all_target_files,args.image_format)
        nn_found = find_nearerst_neighbour(target_image,args.n_value,f_dict)

        effective_result_path = output_dir+"{} most similar images/".format(args.n_value)
        summary_file_name = "findings_summary.txt"
        copy_files_to_dir(effective_result_path,nn_found)
        np.savetxt(effective_result_path+summary_file_name, nn_found, fmt='%s')

        print("Finished processing! You can now go find {} most similar images.\n"
              "Refer the findings_summary.txt file to know the origin of found images".format(args.n_value))

        if(len(m_list) > 0):
            alert = "Please inspect the following files manually as they could not be processed."
            alert = colored(alert, 'red')
            print(alert)
            for file in m_list:
                print(colored(posix_escape(file),'yellow'))
    else:
        print(colored("Unknown error occured during recovery!", 'red'))

def main():
    '''
    This function is the main function of the program. It parses the arguments passed to the program when it it run.
    '''
    parser = argparse.ArgumentParser(description="Sift for similar images in a disk image")
    parser.add_argument("-in",help="Disk image to be searched for" ,dest="input_file", type=str, required=True)
    parser.add_argument("-out",help="Output directory to place all recovered files" ,dest="output_directory", type=str,
                        required=True)
    parser.add_argument("-find", help="Path of target image to be found", dest="image_to_find", type=str, required=True)
    parser.add_argument("-format", help="Image format to be search for(jpg or png). Default = jpg",
                        dest="image_format",type=str, default="jpg")
    parser.add_argument("-n",help="N-most similar images to find." ,dest="n_value", type=int, required=True)
    parser.add_argument("-t", help="[1]. Recover resident files only or \n[2]. Recover resident and deleted"
                                   " files. Defualt = 2", dest="recovery_type",type=int, default="2")
    parser.set_defaults(func = driver)
    args = parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main()

