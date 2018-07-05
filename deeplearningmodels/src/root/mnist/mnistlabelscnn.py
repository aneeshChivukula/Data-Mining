'''
Created on 12 Feb. 2018

@author: 99176493


Deep MNIST Tutorials:
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/
https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5

https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
https://nextjournal.com/a/17592186058848

IWGAN parameters:
nh=8
nw=16
N=128

# https://github.com/igul222/improved_wgan_training/blob/master/tflib/save_images.py
# https://docs.scipy.org/doc/scipy/reference/misc.html
# Load all train and test images in Iwgan : Each x has shape (28,28)

BEGAN, InfoGAN parameters:
nh=8
nw=8
N=64

# Load all test images only in Began and Infogan
# 8*8 image consisting each image 28, 28, 1
# Use same loader method with different input image shapes

# Each image as same flatten order as tensorflow mnist - assuming row-major order which is default for numpy flatten() - check other flatten orders of output labels are incorrect on inspection








cifar10_game:

load_gansamples

binarizermulti
binarizerganmulti
binarizer

trainmultiplayercnn==True
testgenadv==True
multiplayer==True

cifar_train:
max_steps



two label training and testing data available on disk, no need of load_gansamples,
binarizerganmulti in numpy makes bin files for cifar10
binarizermulti creates manipulated data for multiplayer game
binarizer creates manipulated data for twoplayer game

trainmultiplayercnn and multiplayer is not required for training and testing respectively on generated data
no need to use mnist data in the testing process for testgenadv, simply use the train and test split directories created for earlier experiments,

trace control flow in training and testing methods, corresponding flags and parameters in testgenadv code block,
check all parameters set appropriately in cifar_train and cifar_eval, any other methods called in either twoplayer or multiplayer tests,

run this command in workspace, run all experiments in deeplearningmodels_25 and deeplearningmodels_26, testgenadv code block is missing on server,
stdbuf -oL python cifar10_game.py > out-phoenix11-5and8.txt
tail -F out-phoenix11-5and8.txt

alphas are in /data/achivuku/Documents
code in /data/achivuku/workspace
copy data directories like cifar10_22 in /data/achivuku/Desktop

Store data in /scratch
Run code in /data


Updates:
ganimages, ganlabels to be of expected shape
copy remaining part of testgenadv from desktop to server
binarizerganmulti, cifar10_train.train(), binarizermulti, cifar10_eval.evaluate() to work as expected

images shape to be correctly maintained across the methods tested
change IMAGE_SIZE from 28 to 32 to be compatible with cifar10-input.py

'''



'''
Created on 12 Feb. 2018

@author: 99176493


Deep MNIST Tutorials:
https://www.tensorflow.org/versions/r0.12/tutorials/mnist/pros/
https://gist.github.com/saitodev/c4c7a8c83f5aa4a00e93084dd3f848c5

https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
https://nextjournal.com/a/17592186058848

IWGAN parameters:
nh=8
nw=16
N=128

# https://github.com/igul222/improved_wgan_training/blob/master/tflib/save_images.py
# https://docs.scipy.org/doc/scipy/reference/misc.html
# Load all train and test images in Iwgan : Each x has shape (28,28)

BEGAN, InfoGAN parameters:
nh=8
nw=8
N=64

# Load all test images only in Began and Infogan
# 8*8 image consisting each image 28, 28, 1
# Use same loader method with different input image shapes

# Each image as same flatten order as tensorflow mnist - assuming row-major order which is default for numpy flatten() - check other flatten orders of output labels are incorrect on inspection








cifar10_game:

load_gansamples

binarizermulti
binarizerganmulti
binarizer

trainmultiplayercnn==True
testgenadv==True
multiplayer==True

cifar_train:
max_steps



two label training and testing data available on disk, no need of load_gansamples,
binarizerganmulti in numpy makes bin files for cifar10
binarizermulti creates manipulated data for multiplayer game
binarizer creates manipulated data for twoplayer game

trainmultiplayercnn and multiplayer is not required for training and testing respectively on generated data
no need to use mnist data in the testing process for testgenadv, simply use the train and test split directories created for earlier experiments,

trace control flow in training and testing methods, corresponding flags and parameters in testgenadv code block,
check all parameters set appropriately in cifar_train and cifar_eval, any other methods called in either twoplayer or multiplayer tests,

run this command in workspace, run all experiments in deeplearningmodels_25 and deeplearningmodels_26, testgenadv code block is missing on server,
stdbuf -oL python cifar10_game.py > out-phoenix11-5and8.txt
tail -F out-phoenix11-5and8.txt

alphas are in /data/achivuku/Documents
code in /data/achivuku/workspace
copy data directories like cifar10_22 in /data/achivuku/Desktop

Store data in /scratch
Run code in /data


Updates:
ganimages, ganlabels to be of expected shape
copy remaining part of testgenadv from desktop to server
binarizerganmulti, cifar10_train.train(), binarizermulti, cifar10_eval.evaluate() to work as expected

images shape to be correctly maintained across the methods tested
change IMAGE_SIZE from 28 to 32 to be compatible with cifar10-input.py






gan,iwgan,began,infogan, 
main parts of code, overnight runs, 
python deps, data download and code configuration, mnist output, 
use any github repo that works, prefer keras impl, run on gpu term only, 
train for two labels to separate data by digit, check pngs across iterations, check control flow across methods, understand methods logic in github files, 
pickle data load methods, png to bin, check pickle.dump() in imsave() in utils.py of conditional dcgan, np.logical_or() in load_mnist() in model.py, pickle.dump() in save_samples() in model.py, load_samples() in main.py, 
how to create as many images as desired? run as many iterations as needed to generate 5000 images in each class, increase batch size to 500 and take last 10 iterations to collect clear images, use maxiter on github codes, targettting 5000 images per class, use iwgan rather than wgan, stop runs when loss becomes constant and digits are no longer blurred, validation data is used to calculate the loss whereas test data is used to calculate the score for the gan model, we can use data generated during both training and testing for our experiments, 


iwgan
https://github.com/igul222/improved_wgan_training
https://github.com/jiamings/wgan
https://github.com/shekkizh/WassersteinGAN.tensorflow
began
https://github.com/carpedm20/BEGAN-tensorflow
https://github.com/mokemokechicken/keras_BEGAN
https://github.com/Heumi/BEGAN-tensorflow
https://github.com/pbontrager/BEGAN-keras
https://github.com/RuiShu/began
https://github.com/2wins/BEGAN-cntk
https://github.com/znxlwm/pytorch-generative-model-collections

semisupervised mnist gan
https://www.geeksforgeeks.org/machine-learning-recognizing-hand-written-digits-mnist-dataset-set-1/
https://towardsdatascience.com/gan-by-example-using-keras-on-tensorflow-backend-1a6d515a60d0

infogan
https://github.com/openai/InfoGAN
https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/InfoGAN
https://github.com/taimir/infogan-keras
https://github.com/eriklindernoren/Keras-GAN

ebgan vs infogan vs began in tensorflow
https://github.com/buriburisuri/ebgan
https://github.com/hwalsuklee/tensorflow-generative-model-collections
https://github.com/kozistr/Awesome-GANs
https://github.com/kozistr/Awesome-GANs/tree/master/InfoGAN

mnist jpgs
http://study.marearts.com/2015/09/mnist-image-data-jpg-files.html
https://www.kaggle.com/scolianni/mnistasjpg
https://gist.github.com/ischlag/41d15424e7989b936c1609b53edd1390
http://codegists.com/snippet/python/mnist-to-jpgpy_ischlag_python
https://github.com/myleott/mnist_png/blob/master/convert_mnist_to_png.py
http://forums.fast.ai/t/how-to-use-kaggles-mnist-data-with-imageclassifierdata/7653/8


https://github.com/igul222/improved_wgan_training/blob/master/tflib/mnist.py

MNIST Dataset(mnist.pkl.gz) : http://deeplearning.net/tutorial/gettingstarted.html
Each of the three lists is a pair formed from a list of images and a list of class labels for each of the images.

mnist_generator(), generate_image()
BATCH_SIZE = 50 # Batch size
fixed_noise_samples = Generator(128, noise=fixed_noise)

two labels, increase batch size, png to bin postprocessing 
change only improved_wgan_training/tflib/mnist.py to give two-labels-mnist
change improved_wgan_training/gan_mnist.py to increase batch size to 500 and verify Dataset iterator 


classlabels = [4,9]
roundingplaces = 2
BATCH_SIZE = 100
ITERS = 200000

    train_sel_set = numpy.logical_or(train_data_tup[1] == classlabels[0], train_data_tup[1] == classlabels[1])
    train_data = (train_data_tup[0][train_sel_set],train_data_tup[1][train_sel_set])

    dev_sel_set = numpy.logical_or(dev_data_tup[1] == classlabels[0], dev_data_tup[1] == classlabels[1])
    dev_data = (dev_data_tup[0][dev_sel_set],dev_data_tup[1][dev_sel_set])

    test_sel_set = numpy.logical_or(test_data_tup[1] == classlabels[0], test_data_tup[1] == classlabels[1])
    test_data = (test_data_tup[0][test_sel_set],test_data_tup[1][test_sel_set])

images, targets = data
sel_set = numpy.logical_or(targets==classlabels[0], targets==classlabels[1])
images = images[sel_set]
targets = targets[sel_set]
images = images[:int(round(images.shape[0], -roundingplaces))]
targets = targets[:int(round(targets.shape[0], -roundingplaces))]

def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        'samples_{}.png'.format(frame)
    )

def load_samples(CurrDir):
    os.chdir(CurrDir)
    allsamples = []        
    for savedfile in sorted(os.listdir(CurrDir),key=lambda x: int(x.split('_')[-1].split('-')[0])):
        if savedfile.endswith(".pkl"):
            fp = open(savedfile,'rb')
            samples = pickle.load(fp)
            fp.close()
            allsamples.extend(samples)    
    return allsamples

  def save_samples(self,samples,savefile):      
    fp = open(savefile,'wb')
    pickle.dump(samples.astype(np.int32, copy=False),fp)

def imsave(images, labels, size, path, images_file_path, classes_file_path):
  samples = []
  for currdiffimage in images:
      scipy.misc.imsave(tempfilepath, currdiffimage)
      currimage = np.array(list(Image.open(tempfilepath).getdata()))    
      currimage = currimage.reshape((32,32,3)).astype(np.uint8)
      samples.append(currimage)
  pickle.dump(samples,open(images_file_path,'wb'))
  pickle.dump(labels,open(classes_file_path,'wb'))

  image = np.squeeze(merge(images, size))
#  image = images[0]
#  print(images[0])
#  print(image)
#  print(image.shape)
#  print(image[image>1])  
#  sys.exit()
  return scipy.misc.imsave(path, image)

from scipy.misc import imread

CurrDir="/home/achivuku/Desktop/improved_wgan_training-master/"
allsamples = [] 

nh=8
nw=16  
h, w = 28, 28  
N=128

for savedfile in sorted(os.listdir(CurrDir)):
    #savedfile="samples_199999.png"
    save_path=CurrDir + savedfile 
    img = imread(save_path)
    X = []
    for n in xrange(0,N):
      i = n%nw
      j = n/nw  
      x = img[j*h:j*h+h, i*w:i*w+w]
      X.append(x)
    allsamples.extend(X)
# https://github.com/igul222/improved_wgan_training/blob/master/tflib/save_images.py  
# https://docs.scipy.org/doc/scipy/reference/misc.html
# Load all train and test images in Iwgan : Each x has shape (28,28)


CurrDir="/home/achivuku/Desktop/tensorflow-generative-model-collections-master/results/BEGAN_mnist_64_62/"
savedfile="BEGAN_epoch004_test_all_classes.png"
save_path=CurrDir+savedfilematplotlib
img = imread(save_path)
nh=8
nw=8
N=64
X = []
for n in xrange(0,N):
  i = n%nw
  j = n/nw
  x = img[j*h:j*h+h, i*w:i*w+w]
  X.append(x)

import pyplot as plt
plt.imshow(X[0])
#plt.imshow(X[1])
plt.show()

# Load all test images only in Began and Infogan
# 8*8 image consisting each image 28, 28, 1
# Use same loader method with different input image shapes
# Each image as same flatten order as tensorflow mnist - assuming row-major order which is default for numpy flatten() - check other flatten orders of output labels are incorrect on inspection


https://stackoverflow.com/questions/33711985/flattening-a-list-of-numpy-arrays


  
  round num records to lower 00, 0000 - all 0 except first digit, shape to be divisible by batch size, 
assuming atleast 2 zeros with 100 size, atleast 3 digits in the subset list, 
discriminator has to give the class label for generator output? labels on wich generated data is given are known in dcgan, but wgan is an unsupervised method, do we use a classification algorithm to label the data?, need to check D and G implementation and especially G's arguments to see if labels are available, also papers say began&iwgan&infogan are completely unsupervised methods, need to make them conditionally supervised like the (linear concat)dcgan code on mnist data(but not imagenet data) or use classification approach(my preference),
Check for gan papers that are either supervised or semisupervised for including in experiments, We label for attack purpose with CNN as the classifier, We use a two label discriminator whereas GAN uses a probabilistic discriminator telling difference between real and fake data, The complexity of the classifier would depend on the data complexity, We choose CNN instead of SVM to avoid explaining the postprocessing step, 
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py

https://github.com/carpedm20/BEGAN-tensorflow

import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

Copy mnist jpgs into one folder. Overwrite file overlaps. Use only train data. Tw labels at a time.
Trainer : from trainer import Trainer, trainer.train(), trainer.test(), 
get_loader : from data_loader import get_loader
nothing to change in trainer.py, pickle at save_image in test()
config.py : put options --dataset, --batch_size, --input_scale_size if required : --load_path, --data_dir, --test_data_path, --num_worker
--batch_size, --input_scale_size set to default, to avoid memory errors first find code to run on celeba dataset then run on mnist dataset, 

checking alternate implementation in keras, need to convert all images to 64*64 size, the current image size for mnist data is 32*32, check image processing code in eclipse workspace, 
first run create_dataset.py, then training.py, then generate_image.py all containing main method, change parameters for mnist in config.py, keras implementation is much cleaner than tensorflow implementation, get correct versions of the dependencies, 
preprocessing for mnist must match that for celeba
began is slower than wgan, need to increase batch size and run for more iterations once it works, 
need to make links between train data and images? links needed only for celeba in data_loader.py, file extension jpeg instead of jpg harcoded in the loader is causing problems?, need to follow the logic for mnist preprocessing to see correct input given to tensorflow, convert from jpeg to jpg with command mogrify -format jpg *.jpeg in pwd, for many file use find -name '*.jpeg' -print0 | xargs -0 -r mogrify -format jpg and convert *.jpeg.jpg *.jpg after running the shell script convertor.sh, then rm -Rf *.jpeg to remove duplicates, 
with corrected input the mnist data is being generated, no need to check the keras code, : python main.py --dataset=mnist --use_gpu=True --num_worker=10 --sample_per_image=100 

how to speed up execution? cant increase batch size to 100 and increase --num_worker to 10 during training, testing on images ought to be faster on validation data, in paper testing data is used to calculate the inception score between test image and generated image, but we can use generated images during both training and testing, we expect testing images quality to be higher than training images quality, we also expect training time to be in the order of several days,also cannot run multiple runs on terminal without terminating session with gpu, need to create a set of weights for every pair of mnist digits to be used later in generating the data, 

[*] Samples saved: logs/CelebA_0206_183335/0_G.png
[*] Samples saved: logs/CelebA_0206_183335/0_D_real.png
[*] Samples saved: logs/CelebA_0206_183335/0_D_fake.png


https://github.com/mokemokechicken/keras_BEGAN/tree/master/src/began

for ext in ["jpg", "jpeg", "png"]: in data_loader.py

need to convert all images to 64*64 size
began is slower than wgan,
cant increase batch size to 100
validation data is used to calculate the wgan loss whereas test data is used to calculate the began score for the gan model, we can use data generated during both training and testing for our experiments, 
in paper testing data is used to calculate the inception score
we can use generated images during both training and testing - yes
we expect testing images quality to be higher than training images quality
we also expect training time to be in the order of several days
need to create a set of weights for every pair of mnist digits to be used later in generating the data
cant we simply test infogan as part of unsupervised learning. not get into semisupervised learning? the learning objectives and class labels(wrt probability distributions) must be similar for comparitive method and our method?
postpone extensions to gans to a future paper, let reviewer suggest ss gan if it is so relevant, 




The GAN Zoo – Deep Hunt
Read-through_ Wasserstein GAN
GitHub - openai_InfoGAN_ Code for reproducing key results in the paper _InfoGAN_ Interpretable Representation Learning by Information Maximizing Generative Adver
GitHub - GKalliatakis_Delving-deep-into-GANs_ A curated, quasi-exhaustive list of state-of-the-art publications and resources about Generative Adversarial Networ
Generative Models Blog Infogan
DeepLearningImplementations_InfoGAN at master · tdeboissiere_DeepLearningImplementations · GitHub
Deep adversarial learning is finally ready 🚀 and will radically change the game
Crash Course On GANs – Scott Hawley – Development Blog
(Peter) Xi Chen's Academic Website

Code for reproducing key results in the paper "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets" https://arxiv.org/abs/1606.03657
https://github.com/openai/InfoGAN
Keras implementation of InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets
https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/InfoGAN
Implementation of InfoGAN in keras
https://github.com/taimir/infogan-keras

Code accompanying the paper "Wasserstein GAN"
https://github.com/martinarjovsky/WassersteinGAN
For training, an NVIDIA GPU is strongly recommended for speed. CPU is supported but training is very slow.
Code for reproducing experiments in "Improved Training of Wasserstein GANs"
https://github.com/igul222/improved_wgan_training
Tensorflow implementation of Wasserstein GAN - arxiv: https://arxiv.org/abs/1701.07875
https://github.com/shekkizh/WassersteinGAN.tensorflow
Tensorflow Implementation of Wasserstein GAN (and Improved version in wgan_v2)
https://github.com/jiamings/wgan
Wasserstein GAN in Keras
https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html

python3 / Implementation of Google Brain's BEGAN in Tensorflow
https://github.com/Heumi/BEGAN-tensorflow
Boundary equilibrium GAN implementation in Tensorflow
https://github.com/RuiShu/began
Implementation BEGAN([Boundary Equilibrium Generative Adversarial Networks](https://arxiv.org/pdf/1703.10717.pdf)) by Keras.
https://github.com/mokemokechicken/keras_BEGAN
A Keras implementation of the BEGAN Paper
https://github.com/pbontrager/BEGAN-keras
BEGAN-Tensorflow: https://github.com/carpedm20/BEGAN-tensorflow


A list of all named GANs!
https://deephunt.in/the-gan-zoo-79597dc8c347
Deep adversarial learning is finally ready 🚀 and will radically change the game
https://medium.com/intuitionmachine/deep-adversarial-learning-is-finally-ready-and-will-radically-change-the-game-f0cfda7b91d3
Read-through: Wasserstein GAN
https://www.alexirpan.com/2017/02/22/wasserstein-gan.html?
Notes About GAN
http://shaofanlai.com/post/6
Crash Course On GANs
https://drscotthawley.github.io/Crash-Course-On-GANs/
A (Very) Gentle Introduction to Generative Adversarial Networks (a.k.a GANs)
https://www.slideshare.net/ThomasDaSilvaPaula/a-very-gentle-introduction-to-generative-adversarial-networks-aka-gans-71614428
Deep Generative Models
http://www.cs.toronto.edu/~urtasun/courses/CSC2541_Winter17/Deep_generative_models.pdf


Keras implementations of Generative Adversarial Networks.
https://github.com/eriklindernoren/Keras-GAN
Implementation of recent Deep Learning papers
https://github.com/tdeboissiere/DeepLearningImplementations
MNIST Generative Adversarial Model in Keras
https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/

What is new with Wasserstein GAN?
https://www.quora.com/What-is-new-with-Wasserstein-GAN
Walkthrough of Wasserstein Generative Adversarial Networks, May 18, 2017
http://www-users.cs.umn.edu/~fabbr013/projects/wgan.html
Read-through: Wasserstein GAN http://www.alexirpan.com/2017/02/22/wasserstein-gan.html?utm_content=buffer7f258&utm_medium=social&utm_source=twitter.com&utm_campaign=buffer

A curated list of state-of-the-art publications and resources about Generative Adversarial Networks (GANs) and their applications.
http://gkalliatakis.com/blog/delving-deep-into-gans
The classical papers and codes about generative adversarial nets
https://github.com/zhangqianhui/AdversarialNetsPapers
A composable Generative Adversarial Network(GAN) with API and command line tool.
https://github.com/255BITS/HyperGAN




https://github.com/eriklindernoren/Keras-GAN/blob/master/infogan/infogan.py

dependencies not satisfied for openai implementation
DeepLearningImplementations code rep does not seem to generate many batches of images, not sure how to test this repo, 
infogan-keras not designed for generating images
Keras-GAN infogan seems best suited for generating images, it saves 100 images in one picture and runs for 50000 iterations saving images every 50 iterations, decide the point to start saving images by visual inspection and stable accuracy/loss, 
multilabel mnist data being loaded from keras datasets
also running keras began with 500 epochs, store the code with changed config available on server, try to increase batch_size from 16 to 100, num_image from 20 to 100, epochs from 500 to 5000, 
have found the codes for wgan and infogan, if required try more began codes, run 4 jobs on 4 servers at any given time, 
infogan does not give clear digits as output, 
keras began is better implementation than tf began, need to create all 10 digits and more images from keras began, 
find a infogan that coverges to stable solutions, 

use default --batch_size, increase epochs to get more data, expecting good results from epoch 25 onwards, # save training results for every 300 steps
python main.py --dataset mnist --gan_type infoGAN --epoch 1000 --batch_size 64

python main.py --dataset mnist --gan_type BEGAN --epoch 1000 --batch_size 64

currently running 
began and infogan in tensorflow-generative-model-collections-master phoenix24,phoenix17 respectively
iwgan in improved_wgan_training-master atlas3
infogan in Keras-GAN-master phoenix20
pending keras began on phoenix24
if required run iwgan and ebgan also from tensorflow-generative-model-collections-master

iwgan result is satisfactory, the run will go on for 200k iterations, 
keras infogan output to be checked after 100k iterations, 
tf began and infogan output to be checked after 100 epochs - satisfactory









cifar10_game:

load_gansamples

binarizermulti
binarizerganmulti
binarizer

trainmultiplayercnn==True
testgenadv==True
multiplayer==True

cifar_train:
max_steps



two label training and testing data available on disk, no need of load_gansamples, 
binarizerganmulti in numpy makes bin files for cifar10 
binarizermulti creates manipulated data for multiplayer game
binarizer creates manipulated data for twoplayer game

trainmultiplayercnn and multiplayer is not required for training and testing respectively on generated data
no need to use mnist data in the testing process for testgenadv, simply use the train and test split directories created for earlier experiments, 

trace control flow in training and testing methods, corresponding flags and parameters in testgenadv code block, 
check all parameters set appropriately in cifar_train and cifar_eval, any other methods called in either twoplayer or multiplayer tests,

run this command in workspace, run all experiments in deeplearningmodels_25 on atlas3 and deeplearningmodels_26 on phoenix6, testgenadv code block is missing on server, 
stdbuf -oL python cifar10_game.py > out-phoenix11-5and8.txt
tail -F out-phoenix11-5and8.txt

alphas are in /data/achivuku/Documents
code in /data/achivuku/workspace
copy data directories like cifar10_22 in /data/achivuku/Desktop

Store data in /scratch
Run code in /data


Updates:
ganimages, ganlabels to be of expected shape
copy remaining part of testgenadv from desktop to server
binarizerganmulti, cifar10_train.train(), binarizermulti, cifar10_eval.evaluate() to work as expected

images shape to be correctly maintained across the methods tested
change IMAGE_SIZE from 28 to 32 to be compatible with cifar10-input.py





IWGAN

GA
(2,8) 
twoplayer
without mixing testing images, error changes to 0.6814 as expected
('finalresultsmulti', [{'recall': 0.9864, 'fpr': 0.2977, 'f1score': 0.6814, 'precision': 0.5204, 'tpr': 0.9864}])
with testing images, error changes as expected but amount of change is lower
('finalresultsmulti', [{'recall': 0.9894, 'fpr': 0.175, 'f1score': 0.784, 'precision': 0.6492, 'tpr': 0.9894}])

multiplayer
without testing
('finalresultsmulti', [{'recall': 0.9874, 'fpr': 0.2638, 'f1score': 0.707, 'precision': 0.5506, 'tpr': 0.9874}])
with testing
('finalresultsmulti', [{'recall': 0.9885, 'fpr': 0.2284, 'f1score': 0.7361, 'precision': 0.5864, 'tpr': 0.9885}])

(4,9)
twoplayer
('finalresultsmulti', [{'recall': 0.9877, 'fpr': 0.2525, 'f1score': 0.7219, 'precision': 0.5689, 'tpr': 0.9877}])

multiplayer
('finalresultsmulti', [{'recall': 0.9879, 'fpr': 0.2221, 'f1score': 0.7466, 'precision': 0.6, 'tpr': 0.9879}])

(1,4)
twoplayer
('finalresultsmulti', [{'recall': 0.9909, 'fpr': 0.3474, 'f1score': 0.631, 'precision': 0.4629, 'tpr': 0.9909}])

multiplayer
('finalresultsmulti', [{'recall': 0.9911, 'fpr': 0.3469, 'f1score': 0.6314, 'precision': 0.4632, 'tpr': 0.9911}])

(5,8)
twoplayer
('finalresultsmulti', [{'recall': 0.9834, 'fpr': 0.3197, 'f1score': 0.679, 'precision': 0.5185, 'tpr': 0.9834}])

multiplayer
('finalresultsmulti', [{'recall': 0.9873, 'fpr': 0.3227, 'f1score': 0.6789, 'precision': 0.5173, 'tpr': 0.9873}])


[3,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9831, 'fpr': 0.3386, 'f1score': 0.6484, 'precision': 0.4837, 'tpr': 0.9831}])

multiplayer
('finalresultsmulti', [{'recall': 0.9832, 'fpr': 0.3323, 'f1score': 0.6526, 'precision': 0.4884, 'tpr': 0.9832}])


[7,9]
twoplayer
('finalresultsmulti', [{'recall': 0.987, 'fpr': 0.1871, 'f1score': 0.7686, 'precision': 0.6293, 'tpr': 0.987}])
multiplayer
('finalresultsmulti', [{'recall': 0.988, 'fpr': 0.1359, 'f1score': 0.8198, 'precision': 0.7006, 'tpr': 0.988}])



[6,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9886, 'fpr': 0.2515, 'f1score': 0.7197, 'precision': 0.5659, 'tpr': 0.9886}])
multiplayer
('finalresultsmulti', [{'recall': 0.9881, 'fpr': 0.2778, 'f1score': 0.6992, 'precision': 0.541, 'tpr': 0.9881}])


[2,6]
twoplayer
('finalresultsmulti', [{'recall': 0.9971, 'fpr': 0.0263, 'f1score': 0.9602, 'precision': 0.9258, 'tpr': 0.9971}])

multiplayer
('finalresultsmulti', [{'recall': 0.9951, 'fpr': 0.0418, 'f1score': 0.9379, 'precision': 0.8869, 'tpr': 0.9951}])







SA
(2,8) 
twoplayer
('finalresultsmulti', [{'recall': 0.9867, 'fpr': 0.3363, 'f1score': 0.6548, 'precision': 0.49, 'tpr': 0.9867}])

multiplayer
('finalresultsmulti', [{'recall': 0.9854, 'fpr': 0.3919, 'f1score': 0.6193, 'precision': 0.4515, 'tpr': 0.9854}])

(4,9)
twoplayer
('finalresultsmulti', [{'recall': 0.9891, 'fpr': 0.1912, 'f1score': 0.7739, 'precision': 0.6356, 'tpr': 0.9891}])

multiplayer
('finalresultsmulti', [{'recall': 0.9879, 'fpr': 0.2317, 'f1score': 0.7387, 'precision': 0.5899, 'tpr': 0.9879}])

(1,4)
twoplayer
('finalresultsmulti', [{'recall': 0.9909, 'fpr': 0.346, 'f1score': 0.6319, 'precision': 0.4639, 'tpr': 0.9909}])
multiplayer
('finalresultsmulti', [{'recall': 0.993, 'fpr': 0.2544, 'f1score': 0.7005, 'precision': 0.5412, 'tpr': 0.993}])


(5,8)
twoplayer
('finalresultsmulti', [{'recall': 0.9831, 'fpr': 0.3254, 'f1score': 0.6751, 'precision': 0.514, 'tpr': 0.9831}])

multiplayer
('finalresultsmulti', [{'recall': 0.984, 'fpr': 0.2961, 'f1score': 0.6956, 'precision': 0.5379, 'tpr': 0.984}])


[3,8]
twoplayer
('finalresultsmulti', [{'recall': 0.983, 'fpr': 0.3394, 'f1score': 0.6479, 'precision': 0.4832, 'tpr': 0.983}])
multiplayer
('finalresultsmulti', [{'recall': 0.9831, 'fpr': 0.3353, 'f1score': 0.6506, 'precision': 0.4862, 'tpr': 0.9831}])


[7,9]
twoplayer
('finalresultsmulti', [{'recall': 0.9894, 'fpr': 0.0994, 'f1score': 0.8611, 'precision': 0.7623, 'tpr': 0.9894}])

multiplayer
('finalresultsmulti', [{'recall': 0.9881, 'fpr': 0.1542, 'f1score': 0.8011, 'precision': 0.6735, 'tpr': 0.9881}])


[6,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9908, 'fpr': 0.1515, 'f1score': 0.8095, 'precision': 0.6843, 'tpr': 0.9908}])

multiplayer
('finalresultsmulti', [{'recall': 0.9897, 'fpr': 0.1845, 'f1score': 0.7775, 'precision': 0.6402, 'tpr': 0.9897}])


[2,6]
twoplayer
('finalresultsmulti', [{'recall': 0.9974, 'fpr': 0.0224, 'f1score': 0.9657, 'precision': 0.9359, 'tpr': 0.9974}])

multiplayer
('finalresultsmulti', [{'recall': 0.9971, 'fpr': 0.0319, 'f1score': 0.9522, 'precision': 0.9112, 'tpr': 0.9971}])





BEGAN
GA
(2,8) 
twoplayer
('finalresultsmulti', [{'recall': 0.9914, 'fpr': 0.1871, 'f1score': 0.7737, 'precision': 0.6343, 'tpr': 0.9914}])
multiplayer
('finalresultsmulti', [{'recall': 0.9912, 'fpr': 0.206, 'f1score': 0.7564, 'precision': 0.6115, 'tpr': 0.9912}])

(4,9)
twoplayer
('finalresultsmulti', [{'recall': 0.9933, 'fpr': 0.1993, 'f1score': 0.7687, 'precision': 0.627, 'tpr': 0.9933}])

multiplayer
('finalresultsmulti', [{'recall': 0.9944, 'fpr': 0.1684, 'f1score': 0.7974, 'precision': 0.6656, 'tpr': 0.9944}])

(1,4)
twoplayer
('finalresultsmulti', [{'recall': 0.9907, 'fpr': 0.3472, 'f1score': 0.631, 'precision': 0.463, 'tpr': 0.9907}])

multiplayer
('finalresultsmulti', [{'recall': 0.9909, 'fpr': 0.3399, 'f1score': 0.636, 'precision': 0.4683, 'tpr': 0.9909}])


(5,8)
twoplayer
('finalresultsmulti', [{'recall': 0.9886, 'fpr': 0.3239, 'f1score': 0.6787, 'precision': 0.5167, 'tpr': 0.9886}])

multiplayer
('finalresultsmulti', [{'recall': 0.9886, 'fpr': 0.3115, 'f1score': 0.6871, 'precision': 0.5265, 'tpr': 0.9886}])

[3,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9912, 'fpr': 0.1685, 'f1score': 0.7887, 'precision': 0.655, 'tpr': 0.9912}])

multiplayer
('finalresultsmulti', [{'recall': 0.9919, 'fpr': 0.1791, 'f1score': 0.7788, 'precision': 0.6411, 'tpr': 0.9919}])

[7,9]
twoplayer
('finalresultsmulti', [{'recall': 0.9903, 'fpr': 0.2809, 'f1score': 0.6919, 'precision': 0.5317, 'tpr': 0.9903}])
multiplayer
('finalresultsmulti', [{'recall': 0.9905, 'fpr': 0.2361, 'f1score': 0.7273, 'precision': 0.5747, 'tpr': 0.9905}])

[6,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9908, 'fpr': 0.1618, 'f1score': 0.7995, 'precision': 0.67, 'tpr': 0.9908}])
multiplayer
('finalresultsmulti', [{'recall': 0.9911, 'fpr': 0.169, 'f1score': 0.7927, 'precision': 0.6604, 'tpr': 0.9911}])
 
[2,6]
twoplayer
('finalresultsmulti', [{'recall': 0.9988, 'fpr': 0.0281, 'f1score': 0.9585, 'precision': 0.9212, 'tpr': 0.9988}])

multiplayer
('finalresultsmulti', [{'recall': 0.9987, 'fpr': 0.0248, 'f1score': 0.9631, 'precision': 0.9299, 'tpr': 0.9987}])


SA
(2,8) 
twoplayer
('finalresultsmulti', [{'recall': 0.9898, 'fpr': 0.2146, 'f1score': 0.7483, 'precision': 0.6015, 'tpr': 0.9898}])

multiplayer
('finalresultsmulti', [{'recall': 0.9896, 'fpr': 0.2526, 'f1score': 0.7166, 'precision': 0.5616, 'tpr': 0.9896}])

(4,9)
twoplayer
('finalresultsmulti', [{'recall': 0.9978, 'fpr': 0.0184, 'f1score': 0.9722, 'precision': 0.948, 'tpr': 0.9978}]) - run again and select best perf
('finalresultsmulti', [{'recall': 0.9978, 'fpr': 0.016, 'f1score': 0.9757, 'precision': 0.9546, 'tpr': 0.9978}])

multiplayer
('finalresultsmulti', [{'recall': 0.9952, 'fpr': 0.131, 'f1score': 0.835, 'precision': 0.7193, 'tpr': 0.9952}])

(1,4)
twoplayer
('finalresultsmulti', [{'recall': 0.9966, 'fpr': 0.0905, 'f1score': 0.8681, 'precision': 0.769, 'tpr': 0.9966}])

multiplayer
('finalresultsmulti', [{'recall': 0.9936, 'fpr': 0.2186, 'f1score': 0.7315, 'precision': 0.5788, 'tpr': 0.9936}])


(5,8)
twoplayer
('finalresultsmulti', [{'recall': 0.9883, 'fpr': 0.3238, 'f1score': 0.6786, 'precision': 0.5166, 'tpr': 0.9883}])
multiplayer
('finalresultsmulti', [{'recall': 0.9888, 'fpr': 0.3105, 'f1score': 0.6879, 'precision': 0.5274, 'tpr': 0.9888}])


[3,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9935, 'fpr': 0.0707, 'f1score': 0.8981, 'precision': 0.8194, 'tpr': 0.9935}])
multiplayer
('finalresultsmulti', [{'recall': 0.9929, 'fpr': 0.1124, 'f1score': 0.8483, 'precision': 0.7404, 'tpr': 0.9929}])

[7,9]
twoplayer
0.651
multiplayer
('finalresultsmulti', [{'recall': 0.9935, 'fpr': 0.1193, 'f1score': 0.8405, 'precision': 0.7283, 'tpr': 0.9935}])



[6,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9888, 'fpr': 0.2697, 'f1score': 0.7057, 'precision': 0.5486, 'tpr': 0.9888}])
multiplayer
('finalresultsmulti', [{'recall': 0.9907, 'fpr': 0.1911, 'f1score': 0.7718, 'precision': 0.6322, 'tpr': 0.9907}])

[2,6]
twoplayer
('finalresultsmulti', [{'recall': 0.999, 'fpr': 0.0054, 'f1score': 0.9914, 'precision': 0.9839, 'tpr': 0.999}])

multiplayer
('finalresultsmulti', [{'recall': 0.9988, 'fpr': 0.0157, 'f1score': 0.9761, 'precision': 0.9543, 'tpr': 0.9988}])



InfoGAN
GA
(2,8) 
twoplayer
('finalresultsmulti', [{'recall': 0.9917, 'fpr': 0.1577, 'f1score': 0.8019, 'precision': 0.6731, 'tpr': 0.9917}])

multiplayer
('finalresultsmulti', [{'recall': 0.9917, 'fpr': 0.1892, 'f1score': 0.7717, 'precision': 0.6316, 'tpr': 0.9917}])

(4,9)
twoplayer
('finalresultsmulti', [{'recall': 0.9892, 'fpr': 0.1602, 'f1score': 0.8029, 'precision': 0.6756, 'tpr': 0.9892}])

multiplayer
('finalresultsmulti', [{'recall': 0.9897, 'fpr': 0.1094, 'f1score': 0.8554, 'precision': 0.7531, 'tpr': 0.9897}])

(1,4)
twoplayer
('finalresultsmulti', [{'recall': 0.9909, 'fpr': 0.3472, 'f1score': 0.6311, 'precision': 0.463, 'tpr': 0.9909}])

multiplayer
('finalresultsmulti', [{'recall': 0.9909, 'fpr': 0.3465, 'f1score': 0.6316, 'precision': 0.4635, 'tpr': 0.9909}])


(5,8)
twoplayer
('finalresultsmulti', [{'recall': 0.9843, 'fpr': 0.3253, 'f1score': 0.6758, 'precision': 0.5145, 'tpr': 0.9843}])

multiplayer
('finalresultsmulti', [{'recall': 0.9851, 'fpr': 0.3048, 'f1score': 0.69, 'precision': 0.5309, 'tpr': 0.9851}])

[3,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9895, 'fpr': 0.2313, 'f1score': 0.7314, 'precision': 0.58, 'tpr': 0.9895}])

multiplayer
('finalresultsmulti', [{'recall': 0.99, 'fpr': 0.2313, 'f1score': 0.7315, 'precision': 0.5801, 'tpr': 0.99}])

[7,9]
twoplayer
('finalresultsmulti', [{'recall': 0.9866, 'fpr': 0.1815, 'f1score': 0.7738, 'precision': 0.6365, 'tpr': 0.9866}])

multiplayer
('finalresultsmulti', [{'recall': 0.9877, 'fpr': 0.1552, 'f1score': 0.7998, 'precision': 0.672, 'tpr': 0.9877}])


[6,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9814, 'fpr': 0.2326, 'f1score': 0.7317, 'precision': 0.5832, 'tpr': 0.9814}])

multiplayer
('finalresultsmulti', [{'recall': 0.981, 'fpr': 0.2489, 'f1score': 0.7183, 'precision': 0.5666, 'tpr': 0.981}])

[2,6]
twoplayer
('finalresultsmulti', [{'recall': 0.9997, 'fpr': 0.0203, 'f1score': 0.9699, 'precision': 0.9418, 'tpr': 0.9997}])
multiplayer
('finalresultsmulti', [{'recall': 0.9996, 'fpr': 0.0273, 'f1score': 0.96, 'precision': 0.9234, 'tpr': 0.9996}])

SA
(2,8) 
twoplayer
('finalresultsmulti', [{'recall': 0.9916, 'fpr': 0.1847, 'f1score': 0.7759, 'precision': 0.6373, 'tpr': 0.9916}])

multiplayer
('finalresultsmulti', [{'recall': 0.992, 'fpr': 0.1706, 'f1score': 0.7894, 'precision': 0.6555, 'tpr': 0.992}])

(4,9)
twoplayer
('finalresultsmulti', [{'recall': 0.992, 'fpr': 0.0097, 'f1score': 0.9818, 'precision': 0.9718, 'tpr': 0.992}])
multiplayer
('finalresultsmulti', [{'recall': 0.9909, 'fpr': 0.0588, 'f1score': 0.9153, 'precision': 0.8504, 'tpr': 0.9909}])

(1,4)
twoplayer
('finalresultsmulti', [{'recall': 0.9911, 'fpr': 0.3431, 'f1score': 0.634, 'precision': 0.4661, 'tpr': 0.9911}])
multiplayer
('finalresultsmulti', [{'recall': 0.9938, 'fpr': 0.2167, 'f1score': 0.7332, 'precision': 0.5809, 'tpr': 0.9938}])

(5,8)
twoplayer
('finalresultsmulti', [{'recall': 0.9842, 'fpr': 0.3253, 'f1score': 0.6757, 'precision': 0.5144, 'tpr': 0.9842}])
multiplayer
('finalresultsmulti', [{'recall': 0.9846, 'fpr': 0.3087, 'f1score': 0.6871, 'precision': 0.5277, 'tpr': 0.9846}])

[3,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9899, 'fpr': 0.2311, 'f1score': 0.7316, 'precision': 0.5803, 'tpr': 0.9899}])

multiplayer
('finalresultsmulti', [{'recall': 0.9898, 'fpr': 0.2312, 'f1score': 0.7316, 'precision': 0.5802, 'tpr': 0.9898}])

[7,9]
twoplayer
('finalresultsmulti', [{'recall': 0.9833, 'fpr': 0.3387, 'f1score': 0.6481, 'precision': 0.4834, 'tpr': 0.9833}])

multiplayer
('finalresultsmulti', [{'recall': 0.9883, 'fpr': 0.1249, 'f1score': 0.8317, 'precision': 0.718, 'tpr': 0.9883}])

[6,8]
twoplayer
('finalresultsmulti', [{'recall': 0.9821, 'fpr': 0.1625, 'f1score': 0.7946, 'precision': 0.6671, 'tpr': 0.9821}])

multiplayer
('finalresultsmulti', [{'recall': 0.9821, 'fpr': 0.1741, 'f1score': 0.7833, 'precision': 0.6514, 'tpr': 0.9821}])


[2,6]
twoplayer
('finalresultsmulti', [{'recall': 0.9999, 'fpr': 0.0127, 'f1score': 0.981, 'precision': 0.9629, 'tpr': 0.9999}])

multiplayer
('finalresultsmulti', [{'recall': 0.9997, 'fpr': 0.0206, 'f1score': 0.9695, 'precision': 0.941, 'tpr': 0.9997}])

need to change train and test data before following runs
test both two player and multiplayer error on same cnn trained once
repeat training for all pairs of digits on each gan data
get results with and without including testing data - showing results only for without testing
get sa as well as ga results at same time 
update results to table in paper, 














'''
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import os, sys
import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import cPickle as pickle

def loadgenerateddata(CurrDir,nh,nw,N):
    h, w = 28, 28
    allsamples = []

    for savedfile in sorted(os.listdir(CurrDir)):
        save_path = CurrDir + savedfile
        img = imread(save_path)

        X = []
        for n in xrange(0, N):
            i = n % nw
            j = n / nw
            x = img[j * h:j * h + h, i * w:i * w + w]
            X.append(x.ravel())
        allsamples.extend(X)

    return np.array(allsamples)

gendatadir = "/home/achivuku/Desktop/tensorflow-generative-model-collections-master/"

# CurrDir = "/home/achivuku/Desktop/improved_wgan_training-master/output/"
# genimagesfile = "iwgan_genimages.pkl"
# genlabelsfile = "iwgan_genlabels.pkl"
# nh, nw, N = 8, 16, 128
# iwgan_testimages = loadgenerateddata(CurrDir, nh, nw, N)

# CurrDir = "/home/achivuku/Desktop/tensorflow-generative-model-collections-master/results/BEGAN_mnist_64_62/output/"
# genimagesfile = "began_genimages.pkl"
# genlabelsfile = "began_genlabels.pkl"
# nh, nw, N = 8, 8, 64
# began_testimages = loadgenerateddata(CurrDir, nh, nw, N)

# print(began_testimages[0].shape)
# plt.imshow(began_testimages[9991].reshape(28,28))
# plt.show()

CurrDir = "/home/achivuku/Desktop/tensorflow-generative-model-collections-master/results/infoGAN_mnist_64_62/output/"
genimagesfile = "infogan_genimages.pkl"
genlabelsfile = "infogan_genlabels.pkl"
nh, nw, N = 8, 8, 64
infogan_testimages = loadgenerateddata(CurrDir, nh, nw, N)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
numgenimages = mnist.train.images.shape[0]

# labels = pickle.load(open(currlabelsfilepath,'rb'))
# print(mnist.train.images.shape[0])
# print(iwgan_testimages.shape)

# print(iwgan_testimages[0:10000])
# print(iwgan_testimages[0])
# print('label',labels[9991])
# plt.imshow(iwgan_testimages[9991].reshape(28,28))
# plt.show()





def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# Input layer
x  = tf.placeholder(tf.float32, [None, 784], name='x')
y_ = tf.placeholder(tf.float32, [None, 10],  name='y_')
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Convolutional layer 1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Convolutional layer 2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer 1
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y')

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

# Training algorithm
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  # max_steps = 10
  max_steps = 10000
  for step in range(max_steps):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if (step % 100) == 0:
      print(step, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
  # print(max_steps, sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
  # print(max_steps, sess.run(y, feed_dict={x: mnist.test.images, keep_prob: 1.0}))
  # labels = sess.run(tf.argmax(y, 1), feed_dict={x: mnist.test.images, keep_prob: 1.0})

  labels = []
  images = []
  for step in range(int(numgenimages / 10000)):
      # currimages = iwgan_testimages[step*10000:(step+1)*10000]
      # currimages = began_testimages[step*10000:(step+1)*10000]
      currimages = infogan_testimages[step*10000:(step+1)*10000]

      currlabels = sess.run(tf.argmax(y, 1), feed_dict={x: currimages, keep_prob: 1.0})
      images.extend(currimages)
      labels.extend(currlabels)
  # print('CNN predicted',len(labels))

  print('currimages.shape',currimages.shape)
  # print('images.shape',images[0:10])
  print('images.shape',len(images))


  pickle.dump(images, open(gendatadir + genimagesfile, 'wb'))
  pickle.dump(labels, open(gendatadir + genlabelsfile, 'wb'))






