# Clever Image Sifter

The purpose of this program is to search for a target image in a disk image smartly, which will combat
anti-forensic technique such as changing one pixel of an image to change the hash of the whole image.

This program recovers all the files on the disk and finds the n-most similar images to the target image.
For the file recovery, sleuthkit is used at the backend. Then, all the files recovered are sifted for the extension type
and all the target images are obtained. For each image obtained, feature vectors are constructed using
pre-trained models from tensorflow hub library. Using these features, Annoy library is used to apply
nearest neighbour algorithm on these images and n-most similar images to the target image is found. Then these found
images are copied to a folder where the user supplied so that they can view it easily.

### Downloading and setting up the tool

You can clone the whole repository and run the install.sh script by running the following command,
```
sh install.sh
```

This should setup all the required tools and libraries to run this tool. Now, if you dont see any errors, everything 
is good to go!

### Tool help

The tool has help section configured. Just run the tool by the following command to find it,
```
./clever-image-sifter.py -h
```

### Example for running the tool
```
./clever-image-sifter.py -in datadump/Pswift01.dd -out datadump/tool_output/ -n 3 -find datadump/Two_chairs.jpg
```
The above command means, find 3 most similar files to datadump/Two_chairs.jpg image in datadump/Pswift01.dd disk image
and place the results of the findings in datadump/tool_output/ folder.

### Known errors

* OSError: SavedModel file does not exist at: /var/folders/yz/9v21_dg122g76q3lbg07qhww0000gn/T/tfhub_modules/e2ca59248fa837fc8308f9ba8711723a88799917/{saved_model.pbtxt|saved_model.pb}

    To get rid of this error, please go the temporary folder pointing in your error and delete the 'tfhub_modules' folder.



