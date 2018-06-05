What is DrivFace?
----------
The DrivFace database contains images sequences of subjects while driving in real scenarios. It is composed of 606 samples of 640—480 pixels each, acquired over different days from 4 drivers (2 women and 2 men) with several facial features like glasses and beard. 

The ground truth contains the annotation of the face bounding box and the facial key points (eyes, nose and mouth). 
A set of labels assigning each image into 3 possible gaze direction classes are given. 

 * The first class is the looking-right class and contains the head angles between -45º and -30º. 
 * The second one is the frontal class and contains the head angles between -15º and 15º. 
 * The last one is the looking-left class and contains the head angles between 30º and 45º. 

The dataset can be downloaded [[here]] (https://archive.ics.uci.edu/ml/machine-learning-databases/00378/DrivFace.zip)

Files and scripts 
--------------------------
DrivImages.zip has the driver images. The image's name has the format: 
* YearMonthDay_subject_Driv_imNum_HeadPose.jpg 

i.e. 20130529_01_Driv_011_f .jpg is a frame of the first driver corresponding to the 11 sequence's image and the head pose is frontal. 
subject = [1:4], imNum = [001:...], HeadPose = lr (looking-right), f (frontal) and lf (looking-left). 

<p align="center">
<img src="docs/images/sample.jpg" width="250">
</p>

drivPoints.txt contains the ground truth in table's format, where the columns have the follow information: 
* fileName is the imagen's name into DrivImages.zip 
* subject = [1:4] 
* imgNum = int 
* label = [1/2/3] (head pose class that corresponding to [lr/f/lf], respectively) 
* ang = [-45, -30/ -15 0 15/ 30 15] (head pose angle) 
* [xF yF wF hF] = face position 
* [xRE yRE] = rigth eye position 
* [xLE yL] = left eye position 
* [xN yN] = Nose position 
* [xRM yRM] = rigth corner of mouth 
* [xLM yLM] = left corner of mouth 

Source
------------------------------

Katerine Diaz-Chito, Aura HernÃ¡ndez-SabatÃ©, Antonio M. LÃ³pez, A reduced feature set for driver head pose estimation, Applied Soft Computing, Volume 45, August 2016, Pages 98-107, ISSN 1568-4946,

Prerequisites
------------

The current version is based on **Python 3.6**. It is recommended to install [anaconda] (https://www.anaconda.com/download/#linux) and create a python 3.6 virtual environment using the below command

```shell
conda create --name drivface python=3.6
source activate drivface
```

Install the requirements using the below command:

```shell
cd DrivFace
pip install -r requirements.txt
```

Download the dataset and Preprocess the data
--------

Run the below python file to download the dataset and preprocess the data. The script downloads the dataset, unzips the files and loads the images as h5py file

```shell
python Preprocess_Data.py
```

Training and Testing
--------

Run the below python file to create a model, train the model using transfer learning and test the model.

```shell
python train.py
```








