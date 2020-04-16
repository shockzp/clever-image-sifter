#%%
#! /usr/bin/env python
import argparse
import tempfile
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import pytsk3 as tsk
import shutil
from termcolor import colored

import tensorflow as tf
import tensorflow_hub as hub
from annoy import AnnoyIndex
import numpy as np

#Global Strings
original_recovery_dir = "Original recovery/"

def posix_escape(input_string):
    return input_string.replace(" ","\\ ")

def get_all_target_files(output_directory,image_format):

    #TO-DO: Get real file format. 1.Scalpel 2.Magic lib 3. Search into archive files 4. Anti-forensic image techniques
    all_target_files = glob.glob(output_directory + original_recovery_dir + '**/*.' + image_format, recursive=True)
    # all_target_files = [posix_escape(item) for item in all_target_files]
    return all_target_files

def move_files_to_target_dir(output_directory,all_files):

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    for file in all_files:
        shutil.copy2(file,output_directory,follow_symlinks=True)

def recover_files(input_file,output_directory,recovery_type):
    print("Starting recovery of files...")
    img = tsk.Img_Info(url=input_file)
    volume = tsk.Volume_Info(img)

    #List of (ID, start address)
    supported_file_systems = ['NTFS', 'FAT']
    file_systems_to_recover=[]
    for part in volume:
        if any(part.desc.decode('utf-8')[0:3] in fs for fs in supported_file_systems):
            file_systems_to_recover.append(part.start)

    #Recover the files
    recovery_flag = "e"
    space_character=" "

    if os.path.exists(output_directory):
        shutil.rmtree(output_directory)
    os.makedirs(output_directory)

    if recovery_type == 1:
        recovery_flag = "a"
    for start_addr in file_systems_to_recover:
        command = "tsk_recover" + space_character + "-" + recovery_flag + "o" + space_character + str(start_addr) + \
                  space_character + input_file + space_character+ output_directory + posix_escape(original_recovery_dir)
        ret = os.system(command)
        if ret != 0:
            print("FATAL ERROR in recovery occured.")
            return False

    print("All files have been recovered! You can find them at {}".format(output_directory + posix_escape(original_recovery_dir)))

    return True

def load_img(image_path):

  img = tf.io.read_file(image_path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  return img

def make_feature_vectors(all_target_files):
    #Load a pretrained model
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    module = hub.load(module_handle)

    #Filename,feature vector
    featurefile_dict={}
    manual_inspection=[]

    for filename in all_target_files:
        try:
            # Pre-process the image
            img = load_img(filename)

            # Calculate the feature vector
            features = module(img)
            feature_set = np.squeeze(features)
            featurefile_dict[filename]=feature_set
        except:
            manual_inspection.append(filename)
            pass

    return featurefile_dict, manual_inspection

def find_nearerst_neighbour(target_image,n_nearest_neighbors,feature_dict):
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

    named_nearest_neighbors=[]
    find_index = file_name_to_file_index[target_image]

    nearest_neighbors = t.get_nns_by_item(find_index, n_nearest_neighbors+1)

    for j in nearest_neighbors:
        named_nearest_neighbors.append(file_index_to_file_name[j])

    named_nearest_neighbors.remove(target_image)
    return named_nearest_neighbors


def create_summary(nn_found,summary_file):
    np.savetxt(summary_file,nn_found,fmt='%s')


def driver(args):
    ret = recover_files(args.input_file,args.output_directory,args.recovery_type)
    if(ret):
        output_dir = posix_escape(args.output_directory)
        all_target_files = get_all_target_files(output_dir, args.image_format)


        target_image = os.path.abspath(posix_escape(args.image_to_find))
        all_target_files.append(target_image)
        print("Started processing files...")
        f_dict,m_list = make_feature_vectors(all_target_files)

        nn_found = find_nearerst_neighbour(target_image,args.n_value,f_dict)
        move_files_to_target_dir(output_dir+"{} most similar images/".format(args.n_value),nn_found)
        #nn_found=['datadump/tool_output/Original recovery/RECYCLER/S-1-5-21-1292428093-2052111302-839522115-1003/Dc1.jpg', 'datadump/tool_output/Original recovery/RECYCLER/S-1-5-21-1292428093-2052111302-839522115-1003/Dc2.jpg', 'datadump/tool_output/Original recovery/Documents and Settings/spike/Local Settings/Temporary Internet Files/Content.IE5/MXMZUR49/out3[1].jpg']
        create_summary(nn_found,output_dir+"{} most similar images/".format(args.n_value)+"findings_summary.txt")
        print("Finished processing! You can now go find {} most similar images.\n Refer the findings_summary.txt file to know their origin".format(args.n_value))
        if(len(m_list)>0):
            alert = "Please inspect the following files manually as they could not be processed."
            alert = colored(alert, 'red')
            print(alert)
            for file in m_list:
                print(colored(posix_escape(file),'yellow'))

def main():
    parser=argparse.ArgumentParser(description="Sift for similar images in a disk image")
    parser.add_argument("-in",help="Disk image to be searched for" ,dest="input_file", type=str, required=True)
    parser.add_argument("-out",help="Output directory to place all recovered files" ,dest="output_directory", type=str, required=True)
    parser.add_argument("-find", help="Path of target image to be found", dest="image_to_find",
                        type=str, required=True)
    parser.add_argument("-format", help="Image format to be search for(jpg or png). Default = jpg", dest="image_format", type=str, default="jpg")
    parser.add_argument("-n",help="N-most similar images to find." ,dest="n_value", type=int, required=True)
    parser.add_argument("-t", help="[1]. Recover resident files only or \n[2]. Recover resident and deleted files. Defualt = 2", dest="recovery_type",type=int, default="2")
    parser.set_defaults(func=driver)
    args=parser.parse_args()
    args.func(args)

if __name__=="__main__":
    main()

