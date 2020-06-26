import scipy.io as sio
import numpy as np
from scipy.optimize import fmin_cobyla
import os
import sys
import tensorflow as tf
import numpy as np
import random
from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel
import time
from tensorflow.python.keras.backend import set_session
from setup_resnet50 import ResnetModel50
from setup_inception import ImageNet_Universal, InceptionModel

import Utils as util

SEED = 121

tf.random.set_random_seed(SEED)
np.random.seed(SEED)


def projection_box_models(a_point, X, Vt, lb, up, eps):
    ## X \in R^{d \times m}
    # d_temp = a_point.size
    X = np.reshape(X, (X.shape[0] * X.shape[1], X.shape[2]))
    min_VtX = np.min(X, axis=0)
    max_VtX = np.max(X, axis=0)

    Lb = np.maximum(-eps, lb - min_VtX)
    Ub = np.minimum(eps, up - max_VtX)
    z_proj_temp = np.clip(a_point, Lb, Ub)
    return z_proj_temp.reshape(a_point.shape)


def function_evaluation_cons_models(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    # x is in [-0.5, 0.5]
    img_vec = x.copy()
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class, = util.model_prediction_u(model, img)
    tmp = orig_prob.copy()
    tmp[:, target_label] = 0
    n = orig_img.shape[0]
    if arg_targeted_attack:  # targeted attack, target_label is false label
        Loss1 = const * np.max(
            [np.log(np.amax(tmp, 1) + 1e-10) - np.log(orig_prob[:, target_label] + 1e-10), [-kappa] * n], 0)
    else:  # untargeted attack, target_label is true label
        Loss1 = const * np.max(
            [np.log(orig_prob[:, target_label] + 1e-10) - np.log(np.amax(tmp, 1) + 1e-10), [-kappa] * n], 0)

    Loss1 = np.mean(Loss1)
    # Loss2 = np.linalg.norm(img[0] - orig_img[0]) ** 2  ### squared norm # check img[0] - orig_img[0],
    return Loss1  # , Loss2


# data, model = MNIST(), MNISTModel("models/mnist", sess, False)
# data, model = CIFAR(), CIFARModel("models/cifar", sess, True)
data = ImageNet_Universal(SEED)
g1 = tf.Graph()
with g1.as_default():
    sess = tf.Session(graph=g1)
    model1 = InceptionModel(sess, True)

g3 = tf.Graph()
with g3.as_default():
    sess = tf.Session(graph=g3)
    model3 = ResnetModel50(sess, True)

models = [model1, model3]

kappa = 0

const = 100
arg_targeted_attack = False
true_classes = [18, 162]
class_n = 20
orig_img, true_label, target_label = [], [], []
class_number = len(true_classes)

for i in range(len(true_classes)):
    # orig_img = np.load('ori_img_backup.npy')
    orig_img_ = data.test_data[np.where(np.argmax(data.test_labels, 1) == true_classes[i])]
    # np.save('ori_img_backup',orig_img)

    # true_label = data.test_labels[np.where(np.argmax(data.test_labels,1) == class_id)]
    _, orig_class1 = util.model_prediction_u(models[0], orig_img_)
    _, orig_class2 = util.model_prediction_u(models[1], orig_img_)
    # filter out the images which misclassified already
    orig_img_ = orig_img_[np.where((orig_class1 == true_classes[i]) & (orig_class2 == true_classes[i]))]
    assert orig_img_.shape[0] >= class_n, 'no enough valid inputs'

    orig_img.append(orig_img_[:class_n])

    # np.save('original_imgsID'+str(class_id), orig_img)
    # true_label = np.zeros((arg_bsz, 1001))
    # true_label[np.arange(arg_bsz), class_id] = 1
    true_label.append(true_classes[i])  # [class_id[i]]*arg_bsz

    target_label.append(true_classes[i])

# orig_img, target = util.generate_data(data, class_id, target_label)
orig_img = np.array(orig_img)
# a = np.load('/home/xukaidi/Workspace/zoadam/original_imgs_ID_18_162_backup.npy')
# assert (a == orig_img).all()
# np.save('original_imgs_ID' + str(true_classes), orig_img)
print('input images shape', orig_img.shape)
print('true label', true_label)
print('target label', target_label)

# orig_img = util.generate_data_classes(data, true_classes, class_n, model)
# # _, true_label, _ = util.model_prediction(model, orig_img)
#
# # true_label = np.zeros((len(true_classes), class_n))
# # for i, l in enumerate(true_classes):
# #     true_label[i, :] = l
#
# target_label = true_classes
# orig_img = orig_img.reshape(len(true_classes), class_n, *orig_img[0].shape)
d = orig_img[0, 0].size

I = 20000
weights = np.ones((len(models), len(true_classes)), dtype=np.float32) * 1.0 / (len(models) * len(true_classes))
total_loss = np.zeros((I,len(models),class_number))
stationary = np.zeros(I)


orig_img_vec = np.resize(orig_img, (len(true_classes), class_n, d))

w_ori_img_vec = orig_img_vec.copy()
w_img_vec = w_ori_img_vec.copy()
delta_adv = np.zeros((1, d))
weights_record, loss_record, time_record = [], [], []

with sess.graph.as_default():
    set_session(sess)
    ########################################################
    # compute graph for F0 minmax

    shape = (1, 299, 299, 3)

    # the variable we're going to optimize over
    # modifier = tf.Variable(np.zeros(shape, dtype=np.float32))
    modifier = tf.placeholder(tf.float32, shape)

    # setup = []
    # setup.append(modifier.assign(assign_modifier))

    # these are variables to be more efficient in sending data to tf
    timg = tf.placeholder(tf.float32, (class_number, class_n, 299, 299, 3))
    tlab = tf.placeholder(tf.float32, (class_number,1, 1001))
    const = tf.placeholder(tf.float32, (len(models), len(true_classes)))

    # and here's what we use to assign them

    # prediction BEFORE-SOFTMAX of the model
    total_grad = 0
    for m in range(len(models)):
        for c in range(class_number):
            output = models[m].predict(modifier + timg[c])

            real = tf.reduce_sum(tlab[c] * output, 1)
            other = tf.reduce_max((1 - tlab[c]) * output - (tlab[c] * 10000), 1)

            loss1 = tf.maximum(0.0, real - other + 0)

            loss1 = tf.reduce_sum(const[m,c] * loss1)
            grad = tf.gradients(loss1, modifier)
            total_grad += grad[0]


    # sess2 = tf.Session()
    # tf.global_variables_initializer().run(session=sess2)
    # model = CIFARModel("models/cifar", sess2, True)
    # sess.run(tf.variables_initializer(var_list=[modifier]))

    # with graph.as_default():
    # set_session(sess)
    tmp_tlab = np.zeros((class_number, 1, 1001))
    for c in range(class_number):
        tmp_tlab[c, :, target_label[c]] = 1

    time_now = time.time()
    for i in range(I):

        for m in range(len(models)):
            for n in range(class_number):
                total_loss[i, m, n] = function_evaluation_cons_models(w_img_vec[n], kappa, target_label[n], 1,
                                                                      models[m], orig_img[n],
                                                                      arg_targeted_attack)

        base_lr = 0.05 / np.sqrt(i + 1)

        #sess.run(setup, feed_dict={assign_modifier: delta_adv.reshape(shape)})
        g = sess.run([total_grad], feed_dict={
            timg: orig_img,
            tlab: tmp_tlab,
            const: weights,
            modifier: delta_adv.reshape(shape)
        })

        d_tmp = delta_adv.copy()
        delta_adv = delta_adv - base_lr * g[0].flatten()
        V_temp = np.ones((1, d))
        # X_temp = orig_img_vec.reshape((-1,1))
        delta_adv = projection_box_models(delta_adv, orig_img_vec, V_temp, -0.5, 0.5, 0.2)
        w_img_vec = w_ori_img_vec + delta_adv

        print(i, total_loss[i], weights)
        weights_record.append(weights)
        w_tmp = weights.copy()
        w_grad = total_loss[i] - 2 * 5 * (weights-1/(len(models)*class_number))  # lmd = 5
        w_proj = weights + 0.01 * w_grad
        weights = util.bisection(w_proj, 1, 1e-5, ub=1e5)
        stationary[i] = util.stationary_gap(d_tmp, delta_adv, base_lr, w_tmp, weights, 0.01)

        time_d = time.time() - time_now
        time_now = time.time()
        time_record.append(time_d)

print(delta_adv.shape)

np.save('loss2_fo_img.npy', np.array(total_loss))
np.save('weights2_fo_img.npy', np.array(weights_record))
np.save('time_record2_fo_img.npy', np.array(time_record))
np.save('stationary_fo_img.npy', np.array(stationary))



del sess
