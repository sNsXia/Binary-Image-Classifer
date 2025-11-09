Title of the work: Binary image classifier

Author: sNsXia

Date created: 11/08/2025



Versions

Python version: 3.10.4

CUDA: 11.2

Cudnn: 8.1

tensorflow: 2.10

numpy: <2
matplotlib: 3.10.7





Description: This is a binary image classifier that uses inception v3 as the base model and trained to classify an image as targeted class or not targeted class (0 or 1). The binary character identifier h5 file includes a model that is trained with human and anime faces (file size about 2GB, around 7000 images each). The binary character classifier.py file imports inception v3 and trains it with the data set to produce a binary output. The test.py file then uses the model trained against test images in the test folder to see if it is anime or human. 



To use this and train is model to classify an image:

1. Upload image data sets of each class into the 'data' folder (one for 'class 1' and one for 'class 2'). The code uses automatic train\_valid split so no validation folder is required
2. In the Run binary\_character\_identifier.py, rename the class names in 'ds\_train\_' and 'ds\_valid' with the class names you wanted (in order 'class1','class2', should be the same order in the folder 'data'). Then, run the binary\_character\_identifier.py with a conda virtual environment to train the model. For versions, follow the versions listed on the top of the file.
3. Put the images you want to classify into the test folder, and in the 'classifier.py' file, rename the list component in 'l\_classes' in the same order as you named them above.
4. Run the classifier.py file, the result should be printed in a list for each image.



Credits and citations: Refer to 'Citation.txt'



