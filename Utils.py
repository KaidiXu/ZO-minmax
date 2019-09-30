from keras.models import Model, model_from_json, Sequential
from PIL import Image

import tensorflow as tf
import os
import numpy as np


def load_AE(codec_prefix, print_summary=False):

    saveFilePrefix = "models/AE_codec/" + codec_prefix + "_"

    decoder_model_filename = saveFilePrefix + "decoder.json"
    decoder_weight_filename = saveFilePrefix + "decoder.h5"

    if not os.path.isfile(decoder_model_filename):
        raise Exception("The file for decoder model does not exist:{}".format(decoder_model_filename))
    json_file = open(decoder_model_filename, 'r')
    decoder = model_from_json(json_file.read(), custom_objects={"tf": tf})
    json_file.close()

    if not os.path.isfile(decoder_weight_filename):
        raise Exception("The file for decoder weights does not exist:{}".format(decoder_weight_filename))
    decoder.load_weights(decoder_weight_filename)

    if print_summary:
        print("Decoder summaries")
        decoder.summary()

    return decoder


def save_img(img, name = "output.png"):

    np.save(name, img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    pic.save(name)

def generate_data(data, id, target_label):
    inputs = []
    target_vec = []

    inputs.append(data.test_data[id])
    target_vec.append(np.eye(data.test_labels.shape[1])[target_label])

    inputs = np.array(inputs)
    target_vec = np.array(target_vec)

    return inputs, target_vec

# def softmax(x):
#     return np.exp(x) / np.exp(x).sum(axis=1)

def model_prediction(model, inputs):
    #logit = model.model.predict(inputs)
    #prob = softmax(logit)
    prob = model.model.predict(inputs)    
    #print("kerker")
    predicted_class = np.argmax(prob)
    prob_str = np.array2string(prob).replace('\n','')
    return prob, predicted_class, prob_str

def model_prediction_u(model, inputs):
    #logit = model.model.predict(inputs)
    #prob = softmax(logit)
    prob = model.model.predict(inputs)    
    #print("kerker")
    predicted_class = np.argmax(prob,1)
    return prob, predicted_class

def bisection(a,eps,xi,ub=1):
    pa = np.clip(a, 0, ub)
    if np.sum(pa) <= eps:
        print('np.sum(pa) <= eps !!!!')
        w = pa
    else:
        mu_l = np.min(a-1)
        mu_u = np.max(a)
        #mu_a = (mu_u + mu_l)/2
        while np.abs(mu_u - mu_l)>xi:
            #print('|mu_u - mu_l|:',np.abs(mu_u - mu_l))
            mu_a = (mu_u + mu_l)/2
            gu = np.sum(np.clip(a-mu_a, 0, ub)) - eps
            gu_l = np.sum(np.clip(a-mu_l, 0, ub)) - eps
            #print('gu:',gu)
            if gu == 0: 
                print('gu == 0 !!!!!')
                break
            if np.sign(gu) == np.sign(gu_l):
                mu_l = mu_a
            else:
                mu_u = mu_a
            
        w = np.clip(a-mu_a, 0, ub)
        
    return w

def stationary_gap(delta1, delta2, alpha, weight1, weight2, beta):
    d = np.linalg.norm(delta1 - delta2) ** 2
    w = np.linalg.norm(weight1 - weight2) ** 2
    #return np.sqrt((1/alpha)*d + (1/beta)*w) 
    return np.sqrt(d + w) 
