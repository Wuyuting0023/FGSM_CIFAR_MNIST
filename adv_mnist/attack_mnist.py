import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2' #ERROR LOG OUTPUT
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image

DIVIDER = '---------------------------'
TRAIN = False
DEBUG_PRINTS = True
BATCH_SZ = 128
DEFAULT_EPOCHS = 15
EPS = 0.02
STEPS = 15


#Define the trainning model
class Model:
	def __init__(self):
		self.x = tf.placeholder(tf.float32, shape=(None,28,28)) #tf.placeholder:define (data type, shape ,name) The input shape is Minist shape
		self.labels = tf.placeholder(tf.int32, shape=(None,)) #label define,
		self.x_flat = tf.layers.Flatten()(self.x) #flatten  Layer1
		self.layer1 = tf.layers.dense(self.x_flat, 128, tf.nn.relu)#fully connected with input from: self.x_flat , Size of output :128 unit , activation =Relu  , Layer2
		self.dropout = tf.layers.dropout(self.layer1, rate=0.20)#tf.layers.dropout(inputs,rate=0.5,noise_shape=None,seed=None,training=False,name=None ) ,  Layer3
		self.logits = tf.layers.dense(self.dropout, 10) #, Layer4
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,logits=self.logits))
#reduce_mean(input_tensor,axis=None,keep_dims=False,name=None,reduction_indices=None) Used to take the mean of Loss
#tf.nn.sparse_softmax_cross_entropy_with_logit(logits, labels, name=None) Used to caculate the probability distrubute diffenence between logits and labels.
		self.train_op = tf.train.AdamOptimizer(0.002).minimize(self.loss)
#tf.train.AdamOptimizer.__init__(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08, use_locking=False, name='Adam')
#tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, gate_gradients=1, aggregation_method=None, colocate_gradients_with_ops=False, name=None, grad_loss=None)
		self.init_op = tf.initializers.global_variables()
#initialize the model global variables
		self.saver = tf.train.Saver()
#Save the model


## FGSM
class AdvModel:
	def __init__(self, model, eps=EPS):
#eps: maximum distortion of adversarial example compared to original input
		self.x_input = tf.placeholder(tf.float32, (None, 28,28))
		self.x_adv = tf.identity(model.x)#tf.identity(input,name=None),
#Return a tensor with the same shape and contents as input.
		self.target_class_input = tf.placeholder(tf.int32, shape=(None,))
# target_class
		self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target_class_input,logits=model.logits)
#tf.nn.sparse_softmax_cross_entropy_with_logit(logits, labels, name=None) Used to caculate the probability distrubute difference between logits and labels.
		self.dydx = tf.gradients(self.cross_entropy, model.x)[0]
#Caculate the Gradient ys with respect to xs
#ys = self.cross_entropy = loss between target and model
#xs = model.x = model input = Mnist
#tf.gradients(ys,xs,grad_ys=None,name='gradients',stop_gradients=None,)
#[0]means the first element
		self.x_adv = self.x_adv - (eps * self.dydx)
#J(θ, x + eps*sign (∇ x J(θ, x, y))
#Update the adv towards the negative gradient
		self.x_adv = tf.clip_by_value(self.x_adv, 0.0, 1.0)
#tf.clip_by_value(t, clip_value_min, clip_value_max, name=None)
#Used to liminate the value , prevent the gradient explosion and gradient lost. 

##DATA pre-process
def load_data():
	mnist = keras.datasets.mnist
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

	train_images = train_images / 255.0
	test_images = test_images / 255.0
         #Normalize the pixel value between 0 to 1.
	if DEBUG_PRINTS:
		print('>>>   MNIST Dataset Info   <<<\n' + DIVIDER)
		print('# of training images:\t%d' % (train_images.shape[0]))
		print('Size of images:\t\t(%d, %d)' % (train_images.shape[1], train_images.shape[2]))

	return train_images, train_labels, test_images, test_labels


def train_model(session, model, timgs, tlabels, epochs=DEFAULT_EPOCHS, batch_sz=BATCH_SZ, save=True):
	def shuffle(a, b):
		p = np.random.permutation(len(a)) #permutation 'a' randomly
#eg. a = [[1, 2], [3, 4],[5,6]] , label [a,b,c] which related to [0,1,2] (row) respectively.  p could be [ 2, 0, 1] randomly.
		return a[p], b[p] 
#returan a[p]=[[5,6],[1,2],[3,4]],b[p] = label [b,a,c]
	for i in range(epochs):
		# Shuffle dataset every epoch
		timg_s, tlabels_s = shuffle(train_images, train_labels) #permutation all train_imges 
		t = int(np.ceil(len(train_images) * 1.0 / batch_sz))  #int: [5.6 → 5], [3.2 → 3] . 
#np.ceil : [-1.8 → -1] ，[-2 → -2], [3.2 → 4] ， batch number = iteration number = [len(train_images) * 1.0 / batch_sz] ,batch_sz = 128
		total_l = 0 #total_l : iteration?
		for j in range(t):  
			start = batch_sz * j
			end = min(batch_sz * j + batch_sz, len(timg_s)) #len(timg_s) : train imgs number  , 
# min(128 * j +128,10000) ,j = iteration number
			_,l = session.run([model.train_op, model.loss], feed_dict={model.x:timg_s[start:end], 
										   model.labels:tlabels_s[start:end]
										   })
#run(fetches, feed_dict=None, options=None, run_metadata=None)  
#feed_dict = replace a with b = {a:b} = 
#replace model.x with timg_s[start:end]  which related to model.train_op and model.loss
#replace   model.labels with tlabels_s[start:end]  which related to model.loss
			total_l += l
		print('Total Loss:\t', total_l)

	save_path = model.saver.save(session, "./tmp/model.ckpt")

def load_model(session, model):
	model.saver.restore(session, "./tmp/model.ckpt")
	return

def evaluate_model(session, model, test_imgs, test_labels):
	correct_pred = tf.argmax(model.logits, 1) # Save the index of array'model.logits' of maximun of Row.
	res = session.run([correct_pred], feed_dict={
										model.x:test_imgs, 
										model.labels:test_labels
										})

#.run:  
#fetches: running the correct_pred, 
#feed_dict = replace model.x with test_imgs, and replace model.label with test_labels.
#input test_imgs and test_labels

	count = 0

	for i in range(len(test_labels)):
		if res[0][i] == test_labels[i]:  #if res maximun probability caculated from test_imgs == test_labels
			count += 1 #The true number ++
	
	return count / len(test_labels) #Return the validation_accuracy

def generate_adv(model, adv, input_img, target_class, num=0, label=1, save=True):
	input_img_arr = [input_img]
	adv_images = [input_img]

	# iterate through model
	for i in range(STEPS):
		adv_images, ls = session.run([adv.x_adv, model.logits], 
			feed_dict={
				model.x: adv_images,
				adv.x_input: input_img_arr,
				adv.target_class_input: [target_class]
		})
#SET model.x with adv_imges;adv.x_input with input_img;adv.target_class_input with target_class
#Return adv_imges and loss between adv_imges and self.label
		# Printing logits
		# if DEBUG_PRINTS:
		# 	print(ls)

	adv_img = adv_images[0]

	# Save image
	if save:
		matplotlib.pyplot.imsave('adv_img_mnist/target_' + str(target_class) + '_base_' + str(label) + '_' + str(num) + '.png', adv_img)

	return adv_img

def setup_session(model):
	sess = tf.Session()
	sess.run(model.init_op)
#self.init_op = tf.initializers.global-variables()
	return sess

# Load MNIST data
train_images, train_labels, test_images, test_labels = load_data()
# Create our basic model
model = Model()

# Create TF session
session = setup_session(model)
# Train the model or load existing weights
if not TRAIN:
	load_model(session, model)
else:
	train_model(session, model, train_images, train_labels)
# Evaluate model
print('Base Accuracy:\t', evaluate_model(session, model, test_images, test_labels) * 100.0, '%')
# Create FSGM method
fgsm = AdvModel(model)

# Let's try and generate some adversarial images to test against
gen_imgs = [] 
gen_labels = []

num = 1
for i in range(len(test_images)): #range in length(num) of test_images
	test_image = test_images[i] #chose the i_th as test_..
	test_label = test_labels[i]

	# Pick random label //
	rand_label = np.random.randint(0,high=10) #define target random from [0,high)

	while rand_label == test_label: #if rand_label = 7 & test_label =7 
		rand_label = np.random.randint(0,high=10) #give rand_label a new number

	adv_img = generate_adv(model, fgsm, test_image, rand_label, num=num, label=test_label, save=	True)
	gen_imgs.append(adv_img) #append:add new word(),eg.(adv_img)
	gen_labels.append(test_label)

	if num % 2000 == 0:
		print('Generated:\t', num + 1, ' images')

	num += 1

# Evaluate against generated samples
print('Model Accuracy Against Adversarial Examples')
print('Adversarial Accuracy:\t', evaluate_model(session, model, gen_imgs, test_labels) * 100, '%')
for i in range(5):
    print(train_labels[i])
#print(model.logits)
