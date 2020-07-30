import os
import errno
import numpy as np
import scipy
import scipy.misc
from glob import glob
import scipy.io as io
import random
import math



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def transform(image, npx=64, is_crop=False, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image,
                                            [resize_w, resize_w])
    return np.array(cropped_image) / 127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])

def save_images_by_name(images, path, names):
    images = inverse_transform(images)
    images = np.squeeze(images)
    mkdir_p(path)

    for i,img in enumerate(images):
        base_name = os.path.basename(names[i])
        save_path = os.path.join(path, base_name)
        scipy.misc.imsave(save_path, img)
        print('Image:{} is saved!'.format(save_path))
        pass

def imread(path, is_grayscale=False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def inverse_transform(image):
    return ((image + 1) * 127.5).astype(np.uint8)

def load_data(image_path, flip=False, is_test=False, image_size = None):
    img = load_image(image_path)
    img = preprocess_img(img, img_size=image_size)
    img = img/127.5 - 1.
    
    if len(img.shape) != 3:
        if len(img.shape)<3:
            _img = np.ones([img.shape[0], img.shape[1], 3])
            _img[:,:,0] = img
            _img[:,:,1] = img
            _img[:,:,2] = img
        elif len(img.shape)>3:
            _img = img[:,:,:3]
        return _img
    else:
        return img

def preprocess_img(img,img_size):
    return scipy.misc.imresize(img, [img_size, img_size])

def load_vectors(fmri_path, flip=False, is_test=False, image_size = 128):
    vctor = load_vector(fmri_path)
    return vctor

def load_vector(fmri_path):
    fmri_ = io.loadmat(fmri_path)
    fmri = fmri_["data_tmp"]
    fmri = np.asarray(fmri)
    return fmri

def load_image(image_path):
    img = imread(image_path)
    img = np.asarray(img)
    return img


class fMRI(object):
    def __init__(self,images_path,phase, batch_size):
        self.dataname = "fMRI"
        self.dims = 100 * 100
        self.shape = [100, 100, 3]
        self.image_size = 100
        self.channel = 3
        self.images_path = images_path
        self.phase=phase
        self.data_list, self.train_lab_list = self.load_fmri(self.phase, batch_size)
    def load_fmri(self, phase, batch_size):
        vector_path = self.images_path+'{}/A/*.mat'.format(phase)
        images_path = self.images_path+'{}/B/*.JPEG'.format(phase)
        files_vector = glob(vector_path)
        files_img = glob(images_path)
        
        leng = len(files_img)
        top = math.ceil(leng/batch_size)
        diff = top * batch_size - leng
        if diff:
            add_img_files = random.sample(files_img, diff)
            add_frmi_files = [name.replace("B", "A").replace("JPEG", "mat") for name in add_img_files ]
            return files_img + add_img_files, files_vector + add_frmi_files
        else:
            return files_img,files_vector