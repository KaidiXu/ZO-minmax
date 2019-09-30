# %load_ext autoreload
# %autoreload 2
import os
import sys
import tensorflow as tf
import numpy as np
import random
import time
from tqdm import tqdm
from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel
#from setup_resnet152 import ResnetModel152
from setup_resnet50 import ResnetModel50
from setup_inception import ImageNet_Universal, InceptionModel


import Utils as util

SEED = 1213
GPUs = 0 # 1 when multiple GPUs are available


args = {} 
#args["minmax"] = 1
args["maxiter"] = 20001
args["init_const"] = 1  ### regularization parameter prior to attack loss
args["dataset"] = "imagenet"

args["kappa"] = 1e-8  ### attack confidence level in attack loss
args["save_iteration"] = False
args["targeted_attack"] = False
args["print_iteration"] = True 
args["decay_lr"] = True
args["exp_code"] = 1
args["lmd"] = 5
args["beta"] = 0.01
### parameter setting for ZO gradient estimation
args["q"] = 10  ### number of random direction vectors
args["mu"] = 0.005  ### key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet

### parameter setting for mini-batch
#args["models_number"] = 2
args["mini_batch_sz"] = 20
args["class_id"] = [18,118]#[18,118,162]
args["target_id"] = 23


args["lr_idx"] = 0
args["lr"] = 0.05
args["constraint"] = 'cons'
#args["mode"] = "ZOPSGD" # ['ZOSMD', 'ZOPSGD', 'ZONES', 'ZOAdaMM']
args["mode"] = "ZOPGD"

def main(args):
    
    
    
    random.seed(SEED)
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    
    print('ZO-minmax case') if MAX_W  else print('ZO-Finite-Sum case')
    args["minmax"] = MAX_W
    class_id = args['class_id'] ### input image (natural example)
    target_id = args['target_id'] ### target images id (adv example) if target attack
    arg_max_iter = args['maxiter'] ### max number of iterations
    arg_init_const = args['init_const'] ### regularization prior to attack loss
    arg_kappa = args['kappa'] ### attack confidence level
    arg_q = args['q'] ### number of random direction vectors
    arg_mode = args['mode'] ### algorithm name
    arg_save_iteration = args['save_iteration']
    arg_Dataset = args["dataset"]
    arg_targeted_attack = args["targeted_attack"]
    #arg_models = args["models_number"]
    arg_bsz = args["mini_batch_sz"]
    idx_lr = args["lr_idx"]
    class_number = len(class_id)

    ## load classofier For MNIST and CIFAR pixel value range is [-0.5,0.5]
    if (arg_Dataset == 'mnist'):
        data, model = MNIST(), MNISTModel("models/mnist", sess, True)
    elif (arg_Dataset == 'cifar10'):
        data, model = CIFAR(), CIFARModel("models/cifar", sess, True)
    elif (arg_Dataset == 'imagenet'):
        data = ImageNet_Universal(SEED)
        
        g1=tf.Graph()
        with g1.as_default():
            if GPUs:
                config1 = tf.ConfigProto(device_count = {'GPU': 0})
                sess1=tf.Session(graph=g1,config=config1)
            else:
                sess1=tf.Session(graph=g1)
            model1 = InceptionModel(sess1, True)
        
#         g2=tf.Graph()
#         with g2.as_default():
#             if GPUs:
#                 config2 = tf.ConfigProto(device_count = {'GPU': 1})
#                 sess2=tf.Session(graph=g2,config=config2)
#             else:
#                 sess2=tf.Session(graph=g2)
#             model2 = ResnetModel152(sess2, True)
#             
        g3=tf.Graph()
        with g3.as_default():
            if GPUs:
                config3 = tf.ConfigProto(device_count = {'GPU': 1})
                sess3=tf.Session(graph=g3,config=config3)
            else:
                sess3=tf.Session(graph=g3)
            model3 = ResnetModel50(sess3, True)
        
        models = [model1, model3]
    else:
        print('Please specify a valid dataset')
        
    # preprocess data for multiple classes
    orig_img, true_label, target_label = [],[],[]
    
    for i in range(len(class_id)):
        
        #orig_img = np.load('ori_img_backup.npy')
        orig_img_ = data.test_data[np.where(np.argmax(data.test_labels,1) == class_id[i])]
        #np.save('ori_img_backup',orig_img)
        
        #true_label = data.test_labels[np.where(np.argmax(data.test_labels,1) == class_id)]
        _, orig_class1  = util.model_prediction_u(models[0],orig_img_) # take 50 or less images to make sure arg_bsz number of them are valid
        _, orig_class2  = util.model_prediction_u(models[1],orig_img_) # take 50 or less images to make sure arg_bsz number of them are valid
        #_, orig_class3  = util.model_prediction_u(models[2],orig_img_) # take 50 or less images to make sure arg_bsz number of them are valid
        # filter out the images which misclassified already
        orig_img_ = orig_img_[np.where((orig_class1 == class_id[i]) & (orig_class2 == class_id[i]))]
        assert orig_img_.shape[0] >= arg_bsz, 'no enough valid inputs'
            
            
        orig_img.append(orig_img_[:arg_bsz])
        
        #np.save('original_imgsID'+str(class_id), orig_img)
        #true_label = np.zeros((arg_bsz, 1001))
        #true_label[np.arange(arg_bsz), class_id] = 1
        true_label.append(class_id[i]) # [class_id[i]]*arg_bsz
        
        if arg_targeted_attack: ### target attack
            #target_label = np.zeros((arg_bsz, 1001))
            #target_label[np.arange(arg_bsz), target_id] = 1
            target_label.append(target_id[i])
        else:
            target_label.append(class_id[i])

    #orig_img, target = util.generate_data(data, class_id, target_label)
    orig_img = np.array(orig_img)
    np.save('original_imgs_ID'+str(class_id), orig_img)
    print('input images shape', orig_img.shape)
    print('true label', true_label)
    print('target label', target_label)
    
    d = orig_img[0,0].size
    print("dimension = ", d)

    # mu=1/d**2  # smoothing parameter
    q = arg_q + 0 
    I = arg_max_iter + 0
    kappa = arg_kappa + 0
    const = arg_init_const + 0


    ## flatten image to vec
    orig_img_vec = np.resize(orig_img, (class_number,arg_bsz, d))

    ## w adv image initialization
    if args["constraint"] == 'uncons':
        # * 0.999999 to avoid +-0.5 return +-infinity 
        w_ori_img_vec = np.arctanh(2 * (orig_img_vec) * 0.999999)  # in real value, note that orig_img_vec in [-0.5, 0.5]
        w_img_vec = w_ori_img_vec.copy()
    else:
        w_ori_img_vec = orig_img_vec.copy()
        w_img_vec = w_ori_img_vec.copy()

    # ## test ##
    # for test_value in w_ori_img_vec[0, :]:
    #     if np.isnan(test_value) or np.isinf(test_value):
    #         print(test_value)


    delta_adv = np.zeros((1,d)) ### initialized adv. perturbation


    # initialize the best solution & best loss
    best_adv_img = []  # successful adv image in [-0.5, 0.5]
    best_delta = []    # best perturbation
    best_distortion = (0.5 * d) ** 2 # threshold for best perturbation
    total_loss = np.zeros((I,len(models),class_number)) ## I: max iters
    l2s_loss_all = np.zeros((I,len(models),class_number))
    stationary = np.zeros(I)
    attack_flag = False
    first_flag = True ## record first successful attack
    weights = np.ones((len(models),class_number),dtype=np.float32)*1.0/(len(models)*class_number)
    weights_record = np.zeros((I,len(models),class_number))
    sr = []
    # parameter setting for ZO gradient estimation
    mu = args["mu"] ### smoothing parameter

    ## learning rate
    base_lr = args["lr"]
    
    if arg_mode == "ZOAdaMM":
        ## parameter initialization for AdaMM
        v_init = 1e-7 #0.00001
        v_hat = v_init * np.ones((1, d))
        v = v_init * np.ones((1, d))

        m = np.zeros((1, d))
        # momentum parameter for first and second order moment
        beta_1 = 0.9
        beta_2 = 0.3  # only used by AMSGrad
        print(beta_1, beta_2)

    #for i in tqdm(range(I)):
    for i in range(I):

        if args["decay_lr"]:
            base_lr = args["lr"]/np.sqrt(i+1)

            
        ## gradient estimation w.r.t. w_img_vec
        if arg_mode == "ZOSCD":
            grad_est = grad_coord_estimation(mu, q, w_img_vec, d, kappa, target_label, const, model, orig_img,
                                             arg_targeted_attack, args["constraint"])
        elif arg_mode == "ZONES":
            grad_est = gradient_estimation_NES(mu, q, w_img_vec, d, kappa, target_label, const, model, orig_img,
                                             arg_targeted_attack, args["constraint"])
        elif args["mode"] == "ZOPGD": # we use weights w instead const here
            grad_est = gradient_estimation_v3(mu, q, w_img_vec, d, kappa, target_label, weights, models, orig_img,
                                               arg_targeted_attack, args["constraint"], class_number)
        else:
            grad_est = gradient_estimation_v2(mu, q, w_img_vec, d, kappa, target_label, const, model, orig_img,
                                               arg_targeted_attack, args["constraint"])

        
        if args["mode"] == "ZOPGD":
            d_tmp = delta_adv.copy()
            delta_adv = delta_adv - base_lr * grad_est
            if args["constraint"] == 'cons':
                #V_temp = np.eye(orig_img_vec.size)
                V_temp = np.ones((1,d))
                #X_temp = orig_img_vec.reshape((-1,1))
                delta_adv = projection_box_models(delta_adv, orig_img_vec, V_temp, -0.5, 0.5, 16/256)
        
        
        # if np.remainder(i,50)==0:
        # print("total loss:",total_loss[i])
        # print(np.linalg.norm(grad_est, np.inf))

        ## ZO-Attack, unconstrained optimization formulation
        if arg_mode == "ZOSGD":
            delta_adv = delta_adv - base_lr * grad_est
        if arg_mode == "ZOsignSGD":
            delta_adv = delta_adv - base_lr * np.sign(grad_est)
        if arg_mode == "ZOSCD":
            delta_adv = delta_adv - base_lr * grad_est
        if arg_mode == "ZOAdaMM":
            m = beta_1 * m + (1-beta_1) * grad_est
            v = beta_2 * v + (1 - beta_2) * np.square(grad_est) ### vt
            #print(np.mean(np.abs(m)),np.mean(np.sqrt(v)))
            v_hat = np.maximum(v_hat,v)
            delta_adv = delta_adv - base_lr * m /np.sqrt(v)
            if args["constraint"] == 'cons':
                tmp = delta_adv.copy()
                #X_temp = orig_img_vec.reshape((-1,1))
                #V_temp2 = np.diag(np.sqrt(v_hat.reshape(-1)+1e-10))
                V_temp = np.sqrt(v_hat.reshape(1,-1))
                delta_adv = projection_box(tmp, orig_img_vec, V_temp, -0.5, 0.5)
                #delta_adv2 = projection_box_2(tmp, X_temp, V_temp2, -0.5, 0.5)
            # v_init = 1e-2 #0.00001
            # v = v_init * np.ones((1, d))
            # m = np.zeros((1, d))
            # # momentum parameter for first and second order moment
            # beta_1 = 0.9
            # beta_2 = 0.99  # only used by AMSGrad
            # m = beta_1 * m + (1-beta_1) * grad_est
            # v = np.maximum(beta_2 * v + (1-beta_2) * np.square(grad_est),v)
            # delta_adv = delta_adv - base_lr * m /np.sqrt(v+1e-10)
            # if args["constraint"] == 'cons':
            #     V_temp = np.diag(np.sqrt(v.reshape(-1)+1e-10))
            #     X_temp = orig_img_vec.reshape((-1,1))
            #     delta_adv = projection_box(delta_adv, X_temp, V_temp, -0.5, 0.5)
        if arg_mode == "ZOSMD":
            delta_adv = delta_adv - 0.5*base_lr * grad_est
            # delta_adv = delta_adv - base_lr* grad_est
            if args["constraint"] == 'cons':
                #V_temp = np.eye(orig_img_vec.size)
                V_temp = np.ones((1,d))
                #X_temp = orig_img_vec.reshape((-1,1))
                delta_adv = projection_box(delta_adv, orig_img_vec, V_temp, -0.5, 0.5)
        if arg_mode == "ZOPSGD":
            delta_adv = delta_adv - base_lr * grad_est
            if args["constraint"] == 'cons':
                #V_temp = np.eye(orig_img_vec.size)
                V_temp = np.ones((1,d))
                #X_temp = orig_img_vec.reshape((-1,1))
                delta_adv = projection_box(delta_adv, orig_img_vec, V_temp, -0.5, 0.5)
        if arg_mode == "ZONES":
            delta_adv = delta_adv - base_lr * np.sign(grad_est)
            if args["constraint"] == 'cons':
                #V_temp = np.eye(orig_img_vec.size)
                V_temp = np.ones((1,d))
                #X = orig_img_vec.reshape((-1,1))
                delta_adv = projection_box(delta_adv, orig_img_vec, V_temp, -0.5, 0.5)

        # if arg_mode == "ZO-AdaFom":
        #     m = beta_1 * m + (1-beta_1) * grad_est
        #     v = v* (float(i)/(i+1)) + np.square(grad_est)/(i+1)
        #     w_img_vec = w_img_vec - base_lr * m/np.sqrt(v)
        ##

        ### adv. example update
        w_img_vec = w_ori_img_vec + delta_adv
        
        ## Total loss evaluation
        if args["constraint"] == 'uncons':
            total_loss[i], l2s_loss_all[i] = function_evaluation_uncons(w_img_vec, kappa, target_label, const, model, orig_img,
                                            arg_targeted_attack)
        else: # we are here
            for m in range(len(models)):
                for n in range(class_number):
                    total_loss[i,m,n] = function_evaluation_cons_models(w_img_vec[n], kappa, target_label[n], const, models[m], orig_img[n],
                                                       arg_targeted_attack)

        # solve max of w here
        if args["mode"] == "ZOPGD":
            if MAX_W:
                w_tmp = weights.copy()
                w_grad = total_loss[i] - 2 * args["lmd"] * (weights-1/(len(models)*class_number))
                w_proj = weights + args["beta"]* w_grad
                weights = util.bisection(w_proj,1,1e-5,ub=1e5)
            weights_record[i] = weights
            
        if MAX_W: stationary[i] = util.stationary_gap(d_tmp, delta_adv, base_lr, w_tmp, weights, args["beta"])
        #print(stationary[i])
        ## covert back to adv_img in [-0.5 , 0.5]
        if args["constraint"] == 'uncons':
            adv_img_vec = 0.5 * np.tanh((w_img_vec)) / 0.999999 # 
        else:
            adv_img_vec = w_img_vec.copy()

        adv_img = np.resize(adv_img_vec, orig_img.shape)
        
        ## print_iteration
        ## update the best solution in the iterations
        #print(weights)
        if args["print_iteration"]:
            if np.remainder(i + 1, 20) == 0:
                for m in range(len(models)):
                    for c in range(class_number):
                        #print('model',m,' class id',class_id[c])
                        attack_prob, _, _ = util.model_prediction(models[m], adv_img[c])
                        target_prob = attack_prob[:,target_label[c]]
                        attack_prob_tmp = attack_prob.copy()
                        attack_prob_tmp[:, target_label[c]] = 0
                        other_prob = np.amax(attack_prob_tmp,1)
                        sr.append(np.sum(true_label[c] != np.argmax(attack_prob,1))/arg_bsz)
                        if (true_label[c] != np.argmax(attack_prob,1)).all():
                            print("model %d class_id %d Iter %d (Succ): ID = %d, lr = %3.7f, decay = %d, ZO = %s %s, loss = %3.5f, TL = %d, PL = %s" % (m, class_id[c], i+1,
                                  class_id[c], args["lr"], int(args["decay_lr"]), arg_mode, args["constraint"], total_loss[i,m,c], true_label[c], np.argmax(attack_prob,1)))
                        else:
                            
                            print("model %d class_id %d Iter %d (Fail): ID = %d, lr = %3.7f, decay = %d, ZO = %s %s, loss = %3.5f, succ rate = %.2f" % (m, class_id[c], i+1,
                                  class_id[c], args["lr"], int(args["decay_lr"]), arg_mode, args["constraint"], total_loss[i,m,c], sr[-1]))
                print(weights)
                #print(np.max(np.abs(delta_adv)),np.min(w_img_vec),np.max(w_img_vec),np.sum(total_loss[i]),)
        print('sum of losses: ',np.sum(total_loss[i]),'weighted loss',np.sum(total_loss[i]*weights))


        if i %1000 == 0 and i !=0 :
            if arg_mode == "ZOAdaMM": print(beta_1, beta_2)
            print("save delta_adv")
            np.save('retimgs_nips/'+str(i)+'itrs'+str(np.argmax(attack_prob,1))+arg_mode+str(args["lr"])+str(args["lmd"]),delta_adv)
            #np.save('retimgs/'+str(i)+'itrs'+str(np.argmax(attack_prob,1))+arg_mode+str(args["lr"])+'_weights',weights_record)
                        
        if arg_save_iteration:
            os.system("mkdir Examples")
            if (np.logical_or(true_label != np.argmax(attack_prob,1), np.remainder(i + 1, 10) == 0)): ## every 10 iterations
                suffix = "id_{}_Mode_{}_True_{}_Pred_{}_Ite_{}".format(class_id, arg_mode, true_label,
                                                                       np.argmax(attack_prob,1), i + 1)
                # util.save_img(adv_img, "Examples/{}.png".format(suffix))

    if (attack_flag):

        ## save data
        suffix0 = "id_{}_Mode_{}_{}_lr_{}_decay_{}_case{}_ini_{}".format(class_id[0], arg_mode, args["constraint"], str(args["lr"]), int(args["decay_lr"]), args["exp_code"], args["init_const"])
        np.savez("{}".format(suffix0), id=class_id, mode=arg_mode, loss=total_loss, weights=weights_record, sr = np.array(sr) ,stationary=stationary
                 #best_distortion=best_distortion, first_distortion=first_distortion,
                 #first_iteration=first_iteration, best_iteation=best_iteration,
                 #learn_rate=args["lr"], decay_lr = args["decay_lr"], attack_flag = attack_flag
                 )
        ## print
        print("It takes {} iteations to find the first attack".format(first_iteration))
        # print(total_loss)
    else:
        ## save data
        suffix0 = "id_{}_Mode_{}_{}_lr_{}_decay_{}_case{}_ini_{}".format(class_id[0], arg_mode, args["constraint"], str(args["lr"]), int(args["decay_lr"]), args["exp_code"], args["init_const"] )
        np.savez("{}".format(suffix0), id=class_id, mode=arg_mode, loss=total_loss, weights=weights_record, sr = np.array(sr) ,stationary=stationary
                 #best_distortion=best_distortion,  learn_rate=args["lr"], decay_lr = args["decay_lr"], attack_flag = attack_flag
                 )
        print("Attack Fails")

    sys.stdout.flush()


# f: objection function
def function_evaluation(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    # x is img_vec format in real value: w
    img_vec = 0.5 * np.tanh(x)/ 0.999999
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class, orig_prob_str = util.model_prediction(model, img)
    tmp = orig_prob.copy()
    tmp[0, target_label] = 0
    if arg_targeted_attack:  # targeted attack
        Loss1 = const * np.max([np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10), -kappa])
    else:  # untargeted attack
        Loss1 = const * np.max([np.log(orig_prob[0, target_label] + 1e-10) - np.log(np.amax(tmp) + 1e-10), -kappa])

    Loss2 = np.linalg.norm(img - orig_img) ** 2
    return Loss1 + Loss2

# f: objection function for unconstrained optimization formulation
def function_evaluation_uncons(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    # x in real value (unconstrained form), img_vec is in [-0.5, 0.5]
    img_vec = 0.5 * np.tanh(x) / 0.999999
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class, orig_prob_str = util.model_prediction(model, img)
    tmp = orig_prob.copy()
    tmp[0, target_label] = 0
    if arg_targeted_attack:  # targeted attack, target_label is false label
        Loss1 = const * np.max([np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10), -kappa])
    else:  # untargeted attack, target_label is true label
        Loss1 = const * np.max([np.log(orig_prob[0, target_label] + 1e-10) - np.log(np.amax(tmp) + 1e-10), -kappa])

    Loss2 = np.linalg.norm(img - orig_img) ** 2
    return Loss1 + Loss2, Loss2

# f: objection function for constrained optimization formulation
# change to universal attack setting
def function_evaluation_cons(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    # x is in [-0.5, 0.5]
    img_vec = x.copy()
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class,  = util.model_prediction_u(model, img)
    tmp = orig_prob.copy()
    tmp[:,target_label] = 0
    n = orig_img.shape[0]
    if arg_targeted_attack:  # targeted attack, target_label is false label
        Loss1 = const * np.max([np.log(np.amax(tmp,1) + 1e-10) - np.log(orig_prob[:,target_label] + 1e-10), [-kappa]*n],0)
    else:  # untargeted attack, target_label is true label
        Loss1 = const * np.max([np.log(orig_prob[:,target_label] + 1e-10) - np.log(np.amax(tmp,1) + 1e-10), [-kappa]*n],0)
    
    Loss1 = np.sum(Loss1)/n
    Loss2 = np.linalg.norm(img[0] - orig_img[0]) ** 2  ### squared norm # check img[0] - orig_img[0], 
    return Loss1 + Loss2, Loss2

def function_evaluation_cons_models(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    # x is in [-0.5, 0.5]
    img_vec = x.copy()
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class,  = util.model_prediction_u(model, img)
    tmp = orig_prob.copy()
    tmp[:,target_label] = 0
    n = orig_img.shape[0]
    if arg_targeted_attack:  # targeted attack, target_label is false label
        Loss1 = const * np.max([np.log(np.amax(tmp,1) + 1e-10) - np.log(orig_prob[:,target_label] + 1e-10), [-kappa]*n],0)
    else:  # untargeted attack, target_label is true label
        Loss1 = const * np.max([np.log(orig_prob[:,target_label] + 1e-10) - np.log(np.amax(tmp,1) + 1e-10), [-kappa]*n],0)
    
    Loss1 = np.mean(Loss1)
    #Loss2 = np.linalg.norm(img[0] - orig_img[0]) ** 2  ### squared norm # check img[0] - orig_img[0], 
    return Loss1#, Loss2

def function_evaluation_cons_models_sum(x, kappa, target_label, weights, models, orig_img, arg_targeted_attack,class_number):
    # x is in [-0.5, 0.5]
    img_vec = x.copy()
    img = np.resize(img_vec, orig_img.shape)
    Loss = 0
    n = orig_img.shape[1]
    for m in range(len(models)):
        for c in range(class_number):
            orig_prob, orig_class,  = util.model_prediction_u(models[m], img[c])
            tmp = orig_prob.copy()
            tmp[:,target_label[c]] = 0
            
            if arg_targeted_attack:  # targeted attack, target_label is false label
                Loss1 = weights[m,c] * np.max([np.log(np.amax(tmp,1) + 1e-10) - np.log(orig_prob[:,target_label[c]] + 1e-10), [-kappa]*n],0)
            else:  # untargeted attack, target_label is true label
                Loss1 = weights[m,c] * np.max([np.log(orig_prob[:,target_label[c]] + 1e-10) - np.log(np.amax(tmp,1) + 1e-10), [-kappa]*n],0)
        
            Loss += np.mean(Loss1)
    return Loss
    
def distortion(a, b):
    return np.linalg.norm(a[0] - b[0]) ### square root


# random directional gradient estimation - averaged over q random directions
def gradient_estimation(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_mode,arg_targeted_attack):
    # x is img_vec format in real value: w
    m, sigma = 0, 100 # mean and standard deviation
    f_0=function_evaluation(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)
    grad_est=0
    for i in range(q):
        u = np.random.normal(m, sigma, (1,d))
        u_norm = np.linalg.norm(u)
        u = u/u_norm
        f_tmp=function_evaluation(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        # gradient estimate
        if arg_mode == "ZO-M-signSGD":
            grad_est=grad_est+ np.sign(u*(f_tmp-f_0))
        else:
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/mu
    return grad_est
    #grad_est=grad_est.reshape(q,d)
    #return d*grad_est.sum(axis=0)/q

def gradient_estimation_v2(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_targeted_attack,arg_cons):
    # x is img_vec format in real value: w
    # m, sigma = 0, 100 # mean and standard deviation
    sigma = 100
    # ## generate random direction vectors
    # U_all_new = np.random.multivariate_normal(np.zeros(d), np.diag(sigma*np.ones(d) + 0), (q,1))


    if arg_cons == 'uncons':
        f_0, ignore =function_evaluation_uncons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)
    else:
        f_0, ignore =function_evaluation_cons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)

    grad_est=0
    for i in range(q):
        u = np.random.normal(0, sigma, (1,d))
        u_norm = np.linalg.norm(u)
        u = u/u_norm
        # ui = U_all_new[i, 0].reshape(-1)
        # u = ui / np.linalg.norm(ui)
        # u = np.resize(u, x.shape)
        if arg_cons == 'uncons':
            ### x+mu*u = x0 + delta + mu*u: unconstrained in R^d, constrained in [-0.5,0.5]^d
            f_tmp, ignore = function_evaluation_uncons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        else:
            f_tmp, ignore = function_evaluation_cons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        # gradient estimate
        # if arg_mode == "ZO-M-signSGD":
        #     grad_est=grad_est+ np.sign(u*(f_tmp-f_0))
        # else:
        grad_est += (d/q)*u*(f_tmp-f_0)/mu
    return grad_est
    #grad_est=grad_est.reshape(q,d)
    #return d*grad_est.sum(axis=0)/q
    
def gradient_estimation_v3(mu,q,x,d,kappa,target_label,const,models,orig_img,arg_targeted_attack,arg_cons,class_number):
    # x is img_vec format in real value: w
    # m, sigma = 0, 100 # mean and standard deviation
    sigma = 100
    # ## generate random direction vectors
    # U_all_new = np.random.multivariate_normal(np.zeros(d), np.diag(sigma*np.ones(d) + 0), (q,1))
    if arg_cons == 'uncons':
        f_0, ignore =function_evaluation_uncons(x,kappa,target_label,const,models,orig_img,arg_targeted_attack)
    else:
        f_0 =function_evaluation_cons_models_sum(x,kappa,target_label,const,models,orig_img,arg_targeted_attack,class_number)

    grad_est=0
    for i in range(q):
        u = np.random.normal(0, sigma, (1,d))
        u_norm = np.linalg.norm(u)
        u = u/u_norm
        # ui = U_all_new[i, 0].reshape(-1)
        # u = ui / np.linalg.norm(ui)
        # u = np.resize(u, x.shape)
        if arg_cons == 'uncons':
            ### x+mu*u = x0 + delta + mu*u: unconstrained in R^d, constrained in [-0.5,0.5]^d
            f_tmp, ignore = function_evaluation_uncons(x+mu*u,kappa,target_label,const,models,orig_img,arg_targeted_attack)
        else:
            f_tmp = function_evaluation_cons_models_sum(x+mu*u,kappa,target_label,const,models,orig_img,arg_targeted_attack,class_number)
        # gradient estimate
        # if arg_mode == "ZO-M-signSGD":
        #     grad_est=grad_est+ np.sign(u*(f_tmp-f_0))
        # else:
        grad_est += (d/q)*u*(f_tmp-f_0)/mu
    return grad_est

def grad_coord_estimation(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_targeted_attack,arg_cons):
    ### q: number of coordinates
    idx_coords_random = np.random.randint(d, size=q) ### note that ZO SCD does not rely on random direction vectors
    grad_coor_ZO = 0
    for id_coord in range(q):
        idx_coord = idx_coords_random[id_coord]
        u = np.zeros(d)
        u[idx_coord] = 1
        u = np.resize(u, x.shape)

        if arg_cons == 'uncons':
            f_old, ignore = function_evaluation_uncons(x-mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            f_new, ignore = function_evaluation_uncons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)

        else:
            f_old, ignore = function_evaluation_cons(x-mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            f_new, ignore = function_evaluation_cons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)

        grad_coor_ZO = grad_coor_ZO + (d / q) * (f_new - f_old) / (2 * mu) * u
    return grad_coor_ZO

def gradient_estimation_NES(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_targeted_attack,arg_cons):
    # x is img_vec format in real value: w
    # m, sigma = 0, 100 # mean and standard deviation
    sigma = 100
    ## generate random direction vectors
    q_prime = int(np.ceil(q/2))
    # U_all_new = np.random.multivariate_normal(np.zeros(d), np.diag(sigma*np.ones(d) + 0), (q_prime,1))


    # if arg_cons == 'uncons':
    #     f_0=function_evaluation_uncons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)
    # else:
    #     f_0=function_evaluation_cons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)

    grad_est=0
    for i in range(q_prime):
        u = np.random.normal(0, sigma, (1,d))
        u_norm = np.linalg.norm(u)
        u = u/u_norm
        # ui = U_all_new[i, 0].reshape(-1)
        # u = ui / np.linalg.norm(ui)
        # u = np.resize(u, x.shape)
        if arg_cons == 'uncons':
            ### x+mu*u = x0 + delta + mu*u: unconstrained in R^d, constrained in [-0.5,0.5]^d
            f_tmp1, ignore = function_evaluation_uncons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            f_tmp2, ignore = function_evaluation_uncons(x-mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        else:
            f_tmp1, ignore = function_evaluation_cons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            f_tmp2, ignore = function_evaluation_cons(x-mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)

        grad_est=grad_est+ (d/q)*u*(f_tmp1-f_tmp2)/(2*mu)
    return grad_est
    #grad_est=grad_est.reshape(q,d)
    #return d*grad_est.sum(axis=0)/q
    
def projection_box_models(a_point, X, Vt, lb, up, eps):
    ## X \in R^{d \times m}
    #d_temp = a_point.size
    X = np.reshape(X, (X.shape[0]*X.shape[1],X.shape[2]))
    min_VtX = np.min(X, axis=0)
    max_VtX = np.max(X, axis=0)

    Lb = np.maximum(-eps, lb - min_VtX)
    Ub = np.minimum( eps, up - max_VtX)
    z_proj_temp = np.clip(a_point,Lb,Ub)
    return z_proj_temp.reshape(a_point.shape)

def projection_box(a_point, X, Vt, lb, up):
    ## X \in R^{d \times m}
    #d_temp = a_point.size
    VtX = np.sqrt(Vt)*X
    
    min_VtX = np.min(VtX, axis=0)
    max_VtX = np.max(VtX, axis=0)

    Lb = lb * np.sqrt(Vt) - min_VtX
    Ub = up * np.sqrt(Vt) - max_VtX
    
    a_temp = np.sqrt(Vt)*a_point
    z_proj_temp = np.multiply(Lb, np.less(a_temp, Lb)) + np.multiply(Ub, np.greater(a_temp, Ub)) \
                  + np.multiply(a_temp, np.multiply( np.greater_equal(a_temp, Lb), np.less_equal(a_temp, Ub)))
    #delta_proj = np.diag(1/np.diag(np.sqrt(Vt)))*z_proj_temp
    delta_proj = 1/np.sqrt(Vt)*z_proj_temp
    #print(delta_proj)
    return delta_proj.reshape(a_point.shape)

def projection_box_2(a_point, X, Vt, lb, up):
    ## X \in R^{d \times m}
    d_temp = a_point.size
    VtX = np.sqrt(Vt)@X

    min_VtX = np.min(VtX, axis=1)
    max_VtX = np.max(VtX, axis=1)

    Lb = lb * np.sqrt(Vt)@np.ones((d_temp,1)) - min_VtX.reshape((-1,1))
    Ub = up * np.sqrt(Vt)@np.ones((d_temp,1)) - max_VtX.reshape((-1,1))
    
    a_temp = np.sqrt(Vt)@(a_point.reshape((-1,1)))
    
    z_proj_temp = np.multiply(Lb, np.less(a_temp, Lb)) + np.multiply(Ub, np.greater(a_temp, Ub)) \
                  + np.multiply(a_temp, np.multiply( np.greater_equal(a_temp, Lb), np.less_equal(a_temp, Ub)))
    
    delta_proj = np.diag(1/np.diag(np.sqrt(Vt)))@z_proj_temp
    #print(delta_proj)
    return delta_proj.reshape(a_point.shape)
### replace inf or -inf in a vector to a finite value
def Inf2finite(a,val_max):
    a_temp = a.reshape(-1)
    for i_temp in range(len(a_temp)):
        test_value = a_temp[i_temp]
        if np.isinf(test_value) and test_value > 0:
            a_temp[i_temp] = val_max
        if np.isinf(test_value) and test_value < 0:
            a_temp[i_temp] = -val_max


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-minmax", "--minmax",type=int, default=1)
    args2 = vars(parser.parse_args())
    MAX_W = args2['minmax']
    main(args)
    
