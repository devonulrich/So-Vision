# So-Vision
COS429 Final Project

## Overview
In this project we attempt to tackle the problem of simultaneous real and virtual face detection. 

## Data:
The datasets that this is run on are the iCartoon Face Detection Dataset: https://drive.google.com/file/d/111cgWh3Z1QBviMMahAGwPKpR3IlNCrsd/view?usp=sharing and Labels: https://drive.google.com/file/d/1qiHHCP1RvMl6kH017pAV8-QDdcMyy8PR/view. In addition, we used the FDDB dataset: http://vis-www.cs.umass.edu/fddb/index.html#download (both the dataset and the labels). Add all these into this directory
 
## Modules:

### Generate Dataset
Uses above datsets to define the combined 50/50 So Vision Dataset. Run by doing running 'python generate-datset.py'

### Baseline
Baseline covers analysis of the pre-trained MTCNN in the python package mtcnn (https://pypi.org/project/mtcnn/) on So Vision dataset. 

### Cartoon-Module
Contains cartoon face detector code as well as analysis code for the Cartoon Face Detector.

### Classifier 
Contains the code for the Binary Classifier

### Combined-Model
Contains code for analysis of Approach 1 and Approach 2

<!-- ## What we have so far (11/26)
In the .ipynb file, I have most of the code for applying face filters. There's two approaches that I tried in there --
an affine transform and a perspective transform. The affine transform is pretty simple & self-explanatory, but it
does a poor job on non-frontal faces. The perspective transform took a bit more work since we need 4 points on the
face, but I did some geometry in the .py file to get the eyes as two points, and two other points parallel to
the mouth.

## To-do List
* ~~Use perspective transform for face filters~~
* Create filter system for left/right mouth keypoints (currently we only have one for left/right eyes)
* Make some more filters! (sunglasses, facial hair, cartoon smile, clown nose, etc)
* Find a dataset to apply filters to

## How to run
I'm using the COS429 enironment that they had us set up. The only other packages that I needed to install were
`mtcnn` and `tensorflow`, so get both through `pip`

The video program can be pretty slow (im getting ~1 FPS), but that's probably still fast enough for us to
process a dataset and add filters to each face. -->
