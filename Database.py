import csv
import os.path as osp
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle
import cv2
    
import PIL
from PIL import Image

# copy a part of the image and rescale to original shape
# used for imitating additional cameras shifted in horizontal direction as compared to the center camera
# img: source image
# scale: value between 0 and 1. portion of the original image to be copied.
# center: tuple (y, x) describing the center point for rescaling the image.
# hpos: value between 0 and 1. shifts the window from left edge (0) to right edge (1) of the original image
def subImg(img, scale, center, hpos):
    shape=np.shape(img)
    #scaled shape
    newshape=(int(shape[0]*scale+0.5), int(shape[1]*scale+0.5), shape[2])
    new_img=np.array(img)
    h=newshape[0]
    w=newshape[1]
    # top border of the sub image
    t=int((1-scale)*center[0])
    # left border of the sub image; shifted by factor hpos
    l=int((1-scale)*center[1])
    l=int(2*l*hpos)
    if (l+w>shape[1]):
        l=l-(l+w-shape[1])
    # create sub image
    new_img=new_img[t:t+h, l:l+w, :]
    new_img=Image.fromarray(new_img)
    # rescale to original size
    new_img=new_img.resize((shape[1], shape[0]), PIL.Image.ANTIALIAS)
    return np.asarray(new_img)

# load database file, corresponding to the final version of the simulator
def loadCSV(rootpath, filename):
    fn=osp.join(rootpath, filename)
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        db=[row for row in reader]
    return [{'left': osp.join(rootpath, row['left']), 
             'right': osp.join(rootpath, row['right']), 
             'center': osp.join(rootpath, row['center']), 
             'steering': float(row['steering']),
             'throttle': float(row['throttle'])} for row in db]

# from database structure as returned by loadCSV, generate input information 
# for training the network consisting of a delta to the steering angle and the file name for the 
# corresponding left, center and right camera (only available in the full version of the simulator).
# explanation for 'delta' can be found in the docu for gen_X_beta, which is used for generating the 
# corresponding input information for the beta version of the simulator without left/right camera
def gen_X(db):
    X=[{'delta': 0.0, 'file': row['left']} for row in db]
    X.extend([{'delta': 0.0, 'file': row['center']} for row in db])
    X.extend([{'delta': 0.0, 'file': row['right']} for row in db])
    return X

# from database structure as returned by loadCSV, generate output information for
# training the network consisting of a steering angle cor the corresponding input image.
# it is possible to apply a bias (which was not used for the final training) and a delta
# steering angle which represents a correction of the steering angle corresponding to 
# the left (+delta) and the right (-delta) camera. Again, this holds for the final version
# of the simulator, NOT for the beta version
# rescale: rescaling the steering angle, by default to the range -0.9 ... 0.9.
# The rescaling is done since I use a tanh activation layer for the output of the network;
# +-1=tanh(x) for x=+-infinity... I wanted to avoid that.
def gen_y(db, delta=0.005, bias=0.0, rescale=0.9):
    y=[row['steering']+delta+bias for row in db]
    y.extend([row['steering']+bias for row in db])
    y.extend([row['steering']-delta+bias for row in db])
    return np.array(y, np.float32)*rescale

# load both X and y data from a CSV database file using the above defined functions.
# if information already is present in X_in, y_in, append those arrays with the added
# content.
def load_Xy(rootpath, filename, X_in=[], y_in=[], delta=0.005, bias=0.001, rescale=0.9):
    db=loadCSV(rootpath, filename)
    X=X_in
    X.extend(gen_X(db))
    y=y_in
    y.extend(gen_y(db, delta, bias, rescale))
    return X, y

# now the same as before, but for the beta version of the simulator. main difference: no
# left/right camera available.

# load database file, corresponding to the beta version of the simulator; so no left/right camera available,
# compare loadCSV
def loadCSV_beta(rootpath, filename):
    fn=osp.join(rootpath, filename)
    with open(fn) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        db=[row for row in reader]
    return [{'center': row['center'], 
             'steering': float(row['steering']),
             'throttle': float(row['throttle'])} for row in db]

# from database structure as returned by loadCSV_beta, generate input information 
# for training the network consisting of a delta to the steering angle and the file name for the 
# corresponding center camera (left/right camera not available here).
# since there is no left/right camera available, the file name is the same for all imitated cameras;
# the corresponding image will be created on the fly during training using the function subImg().
# delta: value to be chose manually, approx. 0.1
# the different values of delta in the structure in the range of -4*delta ... 4*delta are used for 
# two purposes: first, the size of the window to create the sub-image will be scale=1-abs(delta).
# second, the sign of the delta will determine if the window is placed on the left edge (- sign) 
# or the right edge (+ sign) of the original image. Compare getImg() for more details.
# I know, there might be more intuitive solutions to this problem... but it works.
def gen_X_beta(db, delta):
    X=[{'delta': 0.0, 'file': row['center']} for row in db]
    X.extend([{'delta': -1*delta, 'file': row['center']} for row in db])
    X.extend([{'delta': -2*delta, 'file': row['center']} for row in db])
    X.extend([{'delta': -3*delta, 'file': row['center']} for row in db])
    X.extend([{'delta': -4*delta, 'file': row['center']} for row in db])
    X.extend([{'delta': 1*delta, 'file': row['center']} for row in db])
    X.extend([{'delta': 2*delta, 'file': row['center']} for row in db])
    X.extend([{'delta': 3*delta, 'file': row['center']} for row in db])
    X.extend([{'delta': 4*delta, 'file': row['center']} for row in db])
    return X

# from database structure as returned by loadCSV_beta, generate output information 
# for training the network consisting of a steering angle that fits with the corresponding input
# images. As we only have the center camera for the beta version of the simulator, I imitate
# multiple left/right cameras by taking sub images from the original one. 
# delta: factor for a correction to the steering angle
# bias: I could add a bias to the steering angle, wich I did not do for the final training of 
# the network.
# rescale: rescaling the steering angle, by default to the range -0.9 ... 0.9.
# The rescaling is done since I use a tanh activation layer for the output of the network;
# +-1=tanh(x) for x=+-infinity...
# now the different delta corrections to the steering angle are performed because I imply that 
# a sub image from the left corner of the original one imitates a camera to the left of the center
# camera, or, the car being off-center of the street. Hence, I add a positive correction value to
# the steering angle.
def gen_y_beta(db, delta=0.005, bias=0.0, rescale=0.9):
    y=[row['steering']+bias for row in db]
    y.extend([row['steering']+bias+1*delta for row in db])
    y.extend([row['steering']+bias+2.2*delta for row in db])
    y.extend([row['steering']+bias+3.8*delta for row in db])
    y.extend([row['steering']+bias+5.5*delta for row in db])
    y.extend([row['steering']-1*delta+bias for row in db])
    y.extend([row['steering']-2.2*delta+bias for row in db])
    y.extend([row['steering']-3.8*delta+bias for row in db])
    y.extend([row['steering']-5.5*delta+bias for row in db])
    return np.array(y, np.float32)*rescale

# load both X and y data from a CSV database file using the above defined functions. 
# (now for images created with the beta version of the simulator)
# if information already is present in X_in, y_in, append those arrays with the added
# content.
def load_Xy_beta(rootpath, filename, X_in=[], y_in=[], delta=0.005, bias=0.001, rescale=0.9):
    db=loadCSV_beta(rootpath, filename)
    X=X_in
    X.extend(gen_X_beta(db, delta))
    y=y_in
    y.extend(gen_y_beta(db, delta, bias, rescale))
    return X, y
    
# normImg(img): remove the selfie of the car at the lower edge of the image.
# The name of the function is somewhat missleading; before I put the normalization to the 
# network using a lambda layer, I did it here with a min/max normalization.
def normImg(img):
    x=np.array(img[0:135,:,:], np.float32)
    return x
    #mx=np.min(x); Mx=np.max(x);
    #return (x-mx)/(Mx-mx)-0.5

# genImg(imgdata)
# imgdata: structure wiht delta and file (filename) as described by the docu
# for gen_X_beta.
# returns the image corresponding to the file name and the delta by applying the
# sumImg function if necessary
def genImg(imgdata):
    delta=imgdata['delta']
    if delta==0.0:
        # if the delta value is zero, return the image that has been processed by normImg.
        return normImg(mpimg.imread(imgdata['file']))
    else:
        # else load the image
        img=mpimg.imread(imgdata['file'])
        # remove the selfie of the car
        img=img[0:135,:,:]
        # compute the relative size of the sub image
        scale=1-np.abs(delta)
        if delta>0:
            # if delta is positive, take the sub image from the right border of the original one
            subimg=subImg(img, scale, (60, 160), 1.0)
        else:
            # otherwise take it from the left border of the original one
            subimg=subImg(img, scale, (60, 160), 0.0)
        #return subimg
        return normImg(subimg)

# genData(X_in, y_in, batch_size)
# generator for generating the batches for training the network.
# X_in: structure with file names and correction deltas as described in the docu for
# gen_X_beta
# y_in: steering angles
# batch_size: number of images to be processed in a single batch
def genData(X_in, y_in, batch_size=10):
    # once again, shuffle (not really needed since already done outside the function)
    X,y=shuffle(X_in,y_in)
    l=len(y)
    # infinitely generate batches
    while 1:
        nb=0
        # images in the batch
        Xb=[]
        # steering angles in the batch
        yb=[]
        i=0
        # while the counter is below batch size, add more images to the batch
        while nb<batch_size:
            # dont take steering angles above 0.95 into account;
            # originally, steering angles from -25° to 25° correspond to normalized values
            # in the range -1.0 ... 1.0. now, +-25° does rarely occur during driving the car
            # around the trak, therefore I assume this to be an erroneous event that I want to
            # remove from the training data. Additionally, by the procedure described in the docu
            # for gen_y_beta, it can happen that absolute values above 1.0 would be created
            # for the normalized steering angels; which in turn would impose a problem to 
            # the tanh activation layer that I use for the output of the network.
            while np.abs(y[i])>0.95:
                i=i+1
                if i>=l:
                    i=0
                
            #flip image: cv2.flip(img, 1)
            # flip every second image and correspondingly apply a sign flip for the steering 
            # angle.
            if i%2==0:
                Xb.append(genImg(X[i]))
                yb.append(y[i])
            else:
                Xb.append(cv2.flip(genImg(X[i]), 1))
                yb.append(-y[i])
                
            #yield((mpimg.imread(X[i]), y[i]))
            nb=nb+1;
            i=i+1; 
            if i>=l:
                i=0
        yield np.array(Xb, np.float32), np.array(yb, np.float32)
        
    #return (mpimg.imread(filename) for filename in X), y
    