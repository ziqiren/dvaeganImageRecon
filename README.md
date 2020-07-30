# Reconstruct natural images from fMRI using D-VAE-GAN

Example code for the paper "Reconstructing seen image from brain activity by visually-guided cognitive representation and adversarial learning". Method for reconstructing natural images from brain activities (fMRI) need a pretrained VAE/GAN model which is already trained on "unlabeled natural images" (just need image data without fMRI recordings). This can make the networks initialized properly, which is  helpful for producing satisfied reconstruction results. After that, you can use alternative encoding models to encode fMRI signals to latent space for recovering stimuli, or just use encoding model we provide.
The method use a three stage strategy to train generate model mentioned above, we provide optimizer method (contain loss func)for each step in model.py file.


## image reconstruction from fMRI signal  
for the natural image example
- run reconstruct_natural_image.py

for network structure details
- check model.py

for train/test data prepare 
- check utils.py



## for your own data

1. train a vae-gan model on images dataset (reconstruct image just from image, but not involve fMRI), maybe need to collect a new external image dataset (for example, divide out a subset of imagenet)
2. train d-vae-gan model as we described in three stage train strategy part
3. Adapt some parameters in args part in model.py , fine-tune the weights for the loss terms on an isolated data set.
4. You should be able to just run reconstruct_natural_image.py then.


### Requirements

- Anaconda Python 3.6
- Tensorflow or Tensorflow-GPU
- Numpy
- Scipy

### Data

- The unpreprocessed fMRI data: [Deep Image Reconstruction@OpenNeuro](https://openneuro.org/datasets/ds001506)
- detailed preprocessing steps can be found in paper [Shen, Horikawa, Majima, and Kamitani (2019) Deep image reconstruction from human brain activity. PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006633)
- data need to be processed to several pairs (fMRI vector, stimulus image) for train/test, for example: (A_1.mat, B_1.JPEG)

### Model

- You need the following tensorflow model files to run the example scripts .
   [d-vaegan-model](https://drive.google.com/file/d/13vIyrjYvG7uuRsvetD7d6mfuUw65Ew0W/view?usp=sharing)



