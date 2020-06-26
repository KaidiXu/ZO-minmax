from __future__ import print_function
import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16})
plt.switch_backend('agg')  ### running for CCC
import os


#############################################################################
##################### Sub functions for data generation #####################
#############################################################################
def sigmoid_truncated(x):
    if 1.0/(1+np.exp(-x))>0.5:
        return 1
    else:
        return 0

def sigmoid_truncated_vec(x):
    binary_class = np.greater(1.0/(1+np.exp(-x)) ,0.5)*1
    # if 1.0/(1+np.exp(-x))>0.5:
    #     return 1
    # else:
    #     return 0
    return binary_class

def generate_data(n,sigma2,x,noise_std,Data_path,filename):#generate a dataset
    m=len(x) ### dimension of coefficient with the ground truth x
    data=np.zeros((n,m+1)) ### n, # of training samples, m: feature vector dimension, m+1: label dimension
    sigma=np.sqrt(sigma2)
    for i in range(0,n):
        a=np.random.normal(0, sigma, m)
        noise=np.random.normal(0, noise_std)
        c=sigmoid_truncated((a.T).dot(x)+noise)
        data[i][0:m]=a
        data[i][m]=c
    np.savez(Data_path + "/" + filename, data=data)
    return data


def generate_train_and_test_data(train_ratio, poison_ratio, Data_path, filename, shuffle=False):#load generated data, get trainning and testing data
    data=np.load(Data_path + "/" + filename + ".npz")
    data=data['data']
    n=np.shape(data)[0]
    train_n=int(np.round(train_ratio*n))
    train_poison_n = int(np.round(poison_ratio*train_n))
    if shuffle:
        shuffled_indexes=list(range(0,n))
        random.shuffle(shuffled_indexes)
        data=data[shuffled_indexes]
    train_data=data[0:train_n]
    train_poison_data=train_data[0:train_poison_n]
    train_clean_data = train_data[train_poison_n:train_n]
    test_data=data[train_n:n]
    np.savez(Data_path + "/" + "train_4_2_SL_" +"PosRatio_" + str(poison_ratio) + "_" + filename + ".npz",train_data=train_data, train_poison_data = train_poison_data, train_clean_data = train_clean_data)
    np.savez(Data_path + "/" + "test_4_2_SL_" +"PosRatio_" + str(poison_ratio) + "_" + filename + ".npz",test_data=test_data)
    return train_data,train_poison_data,train_clean_data,test_data


def load_train_and_test_data(poison_ratio, Data_path, filename):#load training and testing data
    train_data=np.load(Data_path + "/" + "train_4_2_SL_" + "PosRatio_" + str(poison_ratio) + "_" + filename + ".npz")
    test_data=np.load(Data_path + "/" + "test_4_2_SL_" + "PosRatio_" + str(poison_ratio) + "_" + filename + ".npz")
    return train_data['train_data'],train_data['train_poison_data'],train_data['train_clean_data'], test_data['test_data']

def generate_index(length,b,iter,Data_path,filename): ### general
    index=[]
    for i in range(0,iter):
        temp=np.array(random.sample(range(0,length),b)) ### mini-batch scheme
        index.append(temp)  #### generate a list
    np.savez(Data_path + "/" +"batch_index_train_4_2_SL_" + filename + ".npz",index=index)
    return index

def load_index(Data_path,filename):
    index=np.load(Data_path + "/" + "batch_index_train_4_2_SL_" + filename + ".npz")
    index=index['index']
    return index

#############################################################################
##################### Sub functions for loss & projections #####################
#############################################################################

def loss_function_batch(delta,x,lambda_x, train_poison_data, train_clean_data, index_batch):#compute loss for a batch
    length=np.shape(train_clean_data)[1]
    num_poison= np.shape(train_poison_data)[0]
    index=list(map(int,index_batch))
    a_clean=train_clean_data[index,0:length-1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison,1)),delta.reshape((1,-1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison),axis=0)
    c_clean =train_clean_data[index,length-1]
    c_poison = train_poison_data[:,length-1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))
    h=1.0/(1+np.exp(-a.dot(x)))
    value=( c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)) )/len(c)-lambda_x*np.linalg.norm(x,2)**2
    return value

def loss_function(delta,x,lambda_x, train_poison_data, train_clean_data):#compute loss for a dataset
    length=np.shape(train_clean_data)[1]
    num_poison= np.shape(train_poison_data)[0]
    a_clean=train_clean_data[:,0:length-1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison,1)),delta.reshape((1,-1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison),axis=0)

    c_clean =train_clean_data[:,length-1]
    c_poison = train_poison_data[:,length-1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h=1.0/(1+np.exp(-a.dot(x)))
    value=( c.dot(np.log(h+1e-15))+(1-c).dot(np.log(1-h+1e-15)) )/len(c)-lambda_x*np.linalg.norm(x,2)**2
    return value

def project_inf(x,epsilon):
    x = np.greater(x,epsilon)*epsilon + np.less(x,-epsilon)*(-epsilon) \
    + np.multiply(np.multiply(np.greater_equal(x,-epsilon),np.less_equal(x,epsilon)),x)
    return x

#############################################################################
##################### Sub functions for  ZO algorithm ZO AG #####################
#############################################################################

def ZOPSGA(func,x0,step,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma_dic =5
    flag=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=x_opt+lr*dx
        y_temp=func(x_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
    return x_opt

def ZOPSGD_bounded(func,x0,epsilon,step,lr=0.1,iter=100,Q=10,project=project_inf):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma_dic =5
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma_dic, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q


        x_temp=project(x_opt-lr*dx,epsilon)
        y_temp=func(x_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
    return x_opt

def AG_minmax_batch(func,delta0,x0,index,step,lr,q_rand_num,epsilon,iter,inner_iter=1):
    ### func： loss function
    ### delta0, x0: initial values
    ### index_batch: batch set for clean parts
    ### step: smooothing parameter
    ### lr: learning rate lr=[lr_delta,lr_x]
    ### iter: outer iterations
    ### inner_iter: inner iteration
    delta_opt=delta0
    x_opt=x0
    D_d=len(delta0)
    D_x=len(x0)
    flag=0
    best_f=func(delta0,x0,index[0])
    AG_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        #### record the initial point
        AG_iter_res[i][0:len(delta0)] = delta_opt  ### first D dimension is delta, then x
        AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        def func_deltafixed(x):
            return func(delta_opt,x,index[i])
        fx_pre = func_deltafixed(x_opt)
        x_opt=ZOPSGA(func_deltafixed,x_opt,step[1],lr[1],inner_iter,q_rand_num)
        fx_post = func_deltafixed(x_opt)
        if fx_post < fx_pre:
            print("Warning! Inner Max. Failed! ZO-AG for Min-Max: Iter = %d, obj_pre = %3.4f, obj_post = %3.4f" % (i, fx_pre, fx_post) )



        def func_xfixed(delta):
            return func(delta,x_opt,index[i])
        fdelta_pre = func_xfixed(delta_opt)
        delta_opt=ZOPSGD_bounded(func_xfixed,delta_opt,epsilon,step[0],lr[0],inner_iter,q_rand_num)
        temp_f=func_xfixed(delta_opt)
        if temp_f > fdelta_pre:
            print("Warning! Outer Min. Failed! ZO-AG for Min-Max: Iter = %d, obj_pre = %3.4f, obj_post = %3.4f" % (
            i, fdelta_pre, temp_f))

        # #### did not record the initial point
        # AG_iter_res[i][0:len(delta0)] = delta_opt  ### first D dimension is delta, then x
        # AG_iter_res[i][len(delta0):len(delta0)+len(x0)] = x_opt

        if i%10 == 0:
            print("ZO-AG for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, q = %d, obj = %3.4f" % (i, lr[0], lr[1], q_rand_num, temp_f) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
            if  i > 9000:
                test = 1
        if temp_f<best_f:
            best_f=temp_f
        else:
            flag=flag+1
            #if flag%3==0:
            #    lr[0]=lr[0]*0.98
    return x_opt,AG_iter_res,AG_time

def AG_run_batch(func,delta0,x0,index,epsilon,step,lr,q_rand, iter,inner_iter=1):
    x_opt,AG_iter_res,AG_time=AG_minmax_batch(func,delta0,x0,index,step,lr,q_rand, epsilon,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time


def AG_main_batch_SL(train_poison_data, train_clean_data,  x_ini, delta_ini,  eps_perturb, n_iters, x_gt, index_batch, lambda_x, lr_delta, lr_x, q_rand_num, Data_path, filename=None ):
    n_x = len(x_ini)
    lr_x = np.min([lr_x, 1/np.sqrt(n_x)])
    lr_delta = np.min([lr_delta, 1 / np.sqrt(n_x)])
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_AG(delta,x,index):
        loss=loss_function_batch(delta,x,lambda_x, train_poison_data, train_clean_data, index)
        return loss

    print("##################################################################")
    print("ZO-AG method")
    time_start=time.time()

    ### highlight: lr=[lr_delta,lr_x]
    x_opt,AG_iter_res,AG_time=AG_run_batch(loss_AG,delta0,x0,index_batch,eps_perturb,step=[0.001,0.001],lr=[lr_delta,lr_x],q_rand = q_rand_num , iter=n_iters,inner_iter=1)

    time_end=time.time()
    print('Time cost of ZO-AG:',time_end-time_start,"s")

    if filename==None:
        np.savez(Data_path + "/" +"ZOAG_4_2_SL.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    else:
        np.savez(Data_path + "/" + "ZOAG_4_2_SL_"+filename+".npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return x_opt, AG_iter_res, AG_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]




#############################################################################
##################### Sub functions for  First Order algorithm FO AG #####################
#############################################################################

################### FO gradient computation
def loss_derivative_x_index(delta,x,lambda_x,train_poison_data, train_clean_data,index):
    length=np.shape(train_clean_data)[1]
    num_poison= np.shape(train_poison_data)[0]
    index=list(map(int,index))
    a_clean=train_clean_data[index,0:length-1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison,1)),delta.reshape((1,-1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison))

    c_clean =train_clean_data[index,length-1]
    c_poison = train_poison_data[:,length-1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h_poison = 1.0 / (1 + np.exp(-a_poison.dot(x)))
    h=1.0/(1+np.exp(-a.dot(x)))

    derivative_x = -(((h-c).T).dot(a)).T/len(c)-2*lambda_x*x
    # derivative_delta = -np.sum(h_poison-c_poison)/len(c_poison)*x
    # derivative = np.concatenate(derivative_delta.reshape(-1),derivative_x.reshape(-1))
    #print(derivative)
    return derivative_x

def loss_derivative_x(delta,x,lambda_x,train_poison_data, train_clean_data):
    length=np.shape(train_clean_data)[1]
    num_poison= np.shape(train_poison_data)[0]
    a_clean=train_clean_data[:,0:length-1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison,1)),delta.reshape((1,-1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison))
    c_clean =train_clean_data[:,length-1]
    c_poison = train_poison_data[:,length-1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h_poison = 1.0 / (1 + np.exp(-a_poison.dot(x)))
    h=1.0/(1+np.exp(-a.dot(x)))

    derivative_x = -(((h-c).T).dot(a)).T/len(c)-2*lambda_x*x
    # derivative_delta = -np.sum(h_poison-c_poison)/len(c_poison)*x
    # derivative = np.concatenate(derivative_delta.reshape(-1),derivative_x.reshape(-1))
    #print(derivative)
    return derivative_x

def loss_derivative_delta_index(delta,x,lambda_x,train_poison_data, train_clean_data,index):
    length = np.shape(train_clean_data)[1]
    num_poison = np.shape(train_poison_data)[0]
    index = list(map(int, index))
    a_clean = train_clean_data[index, 0:length - 1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison, 1)), delta.reshape((1, -1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison))

    c_clean = train_clean_data[index, length - 1]
    c_poison = train_poison_data[:, length - 1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h_poison = 1.0 / (1 + np.exp(-a_poison.dot(x)))
    h = 1.0 / (1 + np.exp(-a.dot(x)))

    # derivative_x = -(((h - c).T).dot(a)).T / len(c) - 2 * lambda_x * x
    if len(c_poison) == 0:
        derivative_delta = 0*x
    else:
        derivative_delta = -np.sum(h_poison-c_poison)/len(c_poison)*x
    # derivative = np.concatenate(derivative_delta.reshape(-1),derivative_x.reshape(-1))
    return derivative_delta

def loss_derivative_delta(delta,x,lambda_x,train_poison_data, train_clean_data):
    length = np.shape(train_clean_data)[1]
    num_poison = np.shape(train_poison_data)[0]
    a_clean = train_clean_data[:, 0:length - 1]
    a_poison = train_poison_data[:, 0:length - 1] + np.matmul(np.ones((num_poison, 1)), delta.reshape((1, -1)))
    # a = []
    # a.append(a_poison)
    # a.append(a_clean)
    a = np.concatenate((a_clean,a_poison))

    c_clean = train_clean_data[:, length - 1]
    c_poison = train_poison_data[:, length - 1]
    # c = []
    # c.append(c_poison)
    # c.append(c_clean)
    c = np.concatenate((c_clean,c_poison))

    h_poison = 1.0 / (1 + np.exp(-a_poison.dot(x)))
    h = 1.0 / (1 + np.exp(-a.dot(x)))

    # derivative_x = -(((h - c).T).dot(a)).T / len(c) - 2 * lambda_x * x
    derivative_delta = -np.sum(h_poison-c_poison)/len(c_poison)*x
    # derivative = np.concatenate(derivative_delta.reshape(-1),derivative_x.reshape(-1))
    return derivative_delta


################### FO optimizer
def FO_run_batch(func,train_poison_data, train_clean_data,delta0,x0,index,epsilon,lambda_x,lr,iter=100,project=project_inf):
    lr=np.array(lr)     # lr: learning rate lr=[lr_delta,lr_x]
    FO_iter_res=np.zeros((iter,len(delta0)+len(x0)))
    FO_time=np.zeros(iter)
    D_d=len(delta0)
    D_x=len(x0)
    delta_opt=delta0
    x_opt=x0
    flag1=0
    best_f = func(delta_opt, x_opt, index[0])
    for i in range(0,iter):

        FO_time[i]=time.time()
        FO_iter_res[i][0:D_d]=delta_opt
        FO_iter_res[i][D_d:D_d+D_x]=x_opt


        dx=loss_derivative_x_index(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data,index[i])
        x_opt=x_opt+dx*lr[1]
        # y_temp=func(delta_opt,x_opt,index[i])
        # if y_temp>func(delta_opt,x_opt,index[i]):
        #     x_opt=x_temp

        ddelta=loss_derivative_delta_index(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data,index[i])
        delta_opt=project(delta_opt-ddelta*lr[0],epsilon)

        y_temp=func(delta_opt,x_opt,index[i])
        if i%10 == 0:
            print("FO for Min-Max: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
            print("delta_max=",end="")
            print(max(delta_opt))
            print("delta_min=",end="")
            print(min(delta_opt))
        if y_temp<best_f:
            best_f=y_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
    return x_opt,FO_iter_res,FO_time

#### FO cal
def FO_main_batch_SL(train_poison_data, train_clean_data,  x_ini,delta_ini,  eps_perturb,
                     n_iters, x_gt, index_batch, lambda_x, lr_delta, lr_x, Data_path,  filename=None ):
    n_x = len(x_ini)
    lr_x = np.min([lr_x, 1/np.sqrt(n_x)])
    lr_delta = np.min([lr_delta, 1 / np.sqrt(n_x)])
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_FO(delta,x,index):
        loss=loss_function_batch(delta,x,lambda_x, train_poison_data, train_clean_data, index)
        return loss

    print("##################################################################")
    print("FO-AG method")
    time_start=time.time()

    x_opt,FO_iter_res,FO_time=FO_run_batch(loss_FO,train_poison_data, train_clean_data,delta0,x0,index_batch,
                                           eps_perturb,lambda_x,lr=[lr_delta,lr_x],iter=n_iters,project=project_inf)

    time_end=time.time()
    print('Time cost of FO-AG:',time_end-time_start,"s")

    if filename==None:
        np.savez(Data_path + "/" +"FOAG_4_2_SL.npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    else:
        np.savez(Data_path + "/" +"FOAG_4_2_SL_"+filename+".npz",x_gt=x_gt,FO_iter_res=FO_iter_res,FO_time=FO_time)
    return x_opt, FO_iter_res, FO_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]



#############################################################################
#################### Sub functions for Non-Adv FO approach #############################
#############################################################################

#### non-min-max first order
def FO_nonAdv_run_batch(func,train_poison_data, train_clean_data,x0,index,lambda_x,lr,iter=100):
    lr=np.array(lr)
    BL_iter_res=np.zeros((iter,2*len(x0)))
    BL_time=np.zeros(iter)
    D_x=len(x0)
    D_d=D_x
    delta_opt=np.zeros(D_d)
    x_opt=x0
    best_f=func(x_opt,index[0])
    flag1=0

    for i in range(0,iter):
        BL_time[i]=time.time()
        BL_iter_res[i][0:D_d]=delta_opt
        BL_iter_res[i][D_d:D_d+D_x]=x_opt

        dx=loss_derivative_x_index(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data,index[i])

        x_opt=x_opt+dx*lr[1]
        y_temp=func(x_opt,index[i])
        if y_temp>best_f:
            best_f=y_temp
        else:
            flag1=flag1+1
            #if flag1%3==0:
            #    lr=lr*0.98
        if i%10 == 0:
            print("Non-Adv FO: Iter = %d, lr_delta=%f, lr_x=%f, obj = %3.4f" % (i, lr[0], lr[1], y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))
    return x_opt,BL_iter_res,BL_time


### run
def FO_nonAdv_main_batch_SL(train_poison_data, train_clean_data, x_ini, delta_ini, eps_perturb,
                     n_iters, x_gt, index_batch, lambda_x,  lr_delta, lr_x, Data_path, filename=None):
    n_x = len(x_ini)
    lr_x = np.min([lr_x, 1 / np.sqrt(n_x)])
    lr_delta = np.min([lr_delta, 1 / np.sqrt(n_x)])
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_FO(x, index):
        loss = loss_function_batch(np.zeros(len(delta0)), x, lambda_x, train_poison_data, train_clean_data, index)
        return loss


    print("##################################################################")
    print("FO-nonAdv method")
    time_start = time.time()

    x_opt, FO_nonAdv_iter_res, FO_nonAdv_time = FO_nonAdv_run_batch(loss_FO, train_poison_data, train_clean_data,
                                                                    x0, index_batch,
                                                                     lambda_x, lr=[lr_delta, lr_x], iter=n_iters)

    time_end = time.time()
    print('Time cost of FO-nonAdv:', time_end - time_start, "s")

    if filename == None:
        np.savez(Data_path + "/" +"FOnonAdv_4_2_SL.npz", x_gt=x_gt, FO_nonAdv_res=FO_nonAdv_iter_res, FO_nonAdv_time=FO_nonAdv_time)
    else:
        np.savez(Data_path + "/" +"FOnonAdv_4_2_SL_" + filename + ".npz", x_gt=x_gt, FO_nonAdv_iter_res=FO_nonAdv_iter_res, FO_nonAdv_time=FO_nonAdv_time)
    return x_opt, FO_nonAdv_iter_res, FO_nonAdv_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]


### retrain after obtaining poisoned data using FO solver
def FO_retrain_poison(train_poison_data, train_clean_data, x_ini, delta_opt,
                     n_iters, lambda_x,  lr_x):
    n_x = len(x_ini)
    lr_x = np.min([lr_x, 1 / np.sqrt(n_x)])
    x_opt = x_ini.copy()
    num_train = np.shape(train_clean_data)[0]

    def loss_retrain(x):
        loss = loss_function_batch(delta_opt, x, lambda_x, train_poison_data, train_clean_data, np.arange(0, num_train).astype(int))
        return loss

    best_f=loss_retrain(x_opt)

    retrain_iter_res=np.zeros((n_iters,n_x))

    #### update x
    for i in range(0, n_iters):
        dx=loss_derivative_x_index(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data, np.arange(0, num_train).astype(int))
        x_opt = x_opt+dx*lr_x

        retrain_iter_res[i][:] = x_opt

        y_temp=loss_retrain(x_opt)

        if y_temp>best_f:
            best_f=y_temp
        if i%100 == 0:
            print("Retrain FO: Iter = %d, lr_x=%f, obj = %3.4f" % (i, lr_x, y_temp) )
            print("x_max=",end="")
            print(max(x_opt))
            print("x_min=",end="")
            print(min(x_opt))

        # np.savez("Retrain_4_2_SL_" + filename + ".npz", retrain_iter_res = retrain_iter_res)
    return x_opt ### x_opt, AG_iter_res = [delta_iter, x_iter]


########################################################################################
######################## Subfunctions for Experiments Plot #############################
########################################################################################

########### loss for all data and all obtained variables
def all_loss(iter_res,lambda_x,train_poison_data, train_clean_data,test_data):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_poison_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    all_train_loss=np.zeros(iter)
    # all_test_loss=np.zeros(iter)
    for i in range(0,iter):
        all_train_loss[i]=loss_function(iter_res_delta[i],iter_res_x[i],lambda_x,train_poison_data, train_clean_data)
    return all_train_loss



def acc_for_D(x,data):#compute loss for a dataset
    length=np.shape(data)[1]
    a=data[:,0:length-1]
    c=data[:,length-1]
    # acc=0
    # for i in range(0,np.shape(data)[0]):
    #     if abs(c[i]-sigmoid_truncated((a[i].T).dot(x)))<1e-2:
    #         acc=acc+1
    # acc=acc/np.shape(data)[0]
    predict_results = sigmoid_truncated_vec(a.dot(x))
    acc = 1-np.sum(np.abs(c - predict_results))/np.shape(data)[0]
    return acc

def acc_for_PoisonD(x, delta, poison_data, clean_data):  # compute loss for a dataset
    length = np.shape(clean_data)[1]
    num_poison= np.shape(poison_data)[0]
    a_clean = clean_data[:, 0:length - 1]
    a_poison = poison_data[:, 0:length - 1] + np.matmul(np  .ones((num_poison, 1)), delta.reshape((1, -1)))
    a = np.concatenate((a_clean, a_poison), axis=0)
    c_clean = clean_data[:, length - 1]
    c_poison = poison_data[:, length - 1]
    c = np.concatenate((c_clean, c_poison))
    # acc = 0
    # for i in range(0, np.shape(a)[0]):
    #     if abs(c[i] - sigmoid_truncated((a[i].T).dot(x))) < 1e-2:
    #         acc = acc + 1
    # acc = acc / np.shape(a)[0]
    predict_results = sigmoid_truncated_vec(a.dot(x))
    acc = 1-np.sum(np.abs(c - predict_results))/np.shape(a)[0]
    return acc

def all_acc(iter_res,train_poison_data, train_clean_data,test_data):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    all_train_acc=np.zeros(iter)
    all_test_acc=np.zeros(iter)
    for i in range(0,iter):
        all_test_acc[i]=acc_for_D(iter_res_x[i],test_data)
        ### poisoned training data
        all_train_acc[i] = acc_for_PoisonD(iter_res_x[i], iter_res_delta[i],train_poison_data, train_clean_data)
    return all_train_acc, all_test_acc

def all_acc_retrain(iter_res,train_poison_data, train_clean_data,test_data, lambda_x):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    all_train_acc=np.zeros(iter)
    all_test_acc=np.zeros(iter)
    iters_retrain = 1000
    for i in range(0,iter):

        ### update iter_res_x[i] based on iter_res_delta[i]
        xi_retrain = FO_retrain_poison(train_poison_data, train_clean_data, iter_res_x[i], iter_res_delta[i],
                                             iters_retrain, lambda_x, 0.05)
        all_test_acc[i]=acc_for_D(xi_retrain,test_data)
        ### poisoned training data
        all_train_acc[i] = acc_for_PoisonD(xi_retrain, iter_res_delta[i],train_poison_data, train_clean_data)
    return all_train_acc, all_test_acc


def stationary_condition(iter_res,train_poison_data, train_clean_data,lambda_x,alpha,beta,epsilon):
    iter=np.shape(iter_res)[0]
    D=np.shape(train_data[0])[0]-1
    iter_res_delta=iter_res[:,0:D]
    iter_res_x=iter_res[:,D:2*D]
    G=np.zeros((iter,2*D))
    for i in range(0,iter):
        delta_opt=iter_res_delta[i]
        x_opt=iter_res_x[i]
        G[i][0:D]=-loss_derivative_x(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data)
        # proj_alpha=project_inf(delta_opt-alpha*loss_derivative_delta(delta_opt,x_opt,lambda_x,train_poison_data, train_clean_data),epsilon=epsilon)
        # G[i][D:2*D]=1/alpha*(delta_opt-proj_alpha)
        proj_alpha = project_inf(delta_opt - loss_derivative_delta(delta_opt, x_opt, lambda_x, train_poison_data, train_clean_data),epsilon=epsilon)
        G[i][D:2 * D] = 1 / 1 * (delta_opt - proj_alpha)
    return np.linalg.norm(G,ord=2,axis=1)

def mean_std(data):
    n=len(data)
    iter=len(data[0])
    mean=np.zeros(iter)
    std=np.zeros(iter)
    for i in range(0,iter):
        iter_data=np.zeros(n)
        for j in range(0,n):
            iter_data[j]=np.array(data[j][i])
        mean[i]=np.mean(iter_data)
        std[i]=np.std(iter_data)
    return mean,std

def plot_shaded_logx(x_plot, mean,std,ax_handle):
    # iter=len(mean)
    mean=np.array(mean)
    std=np.array(std)
    low=mean-std
    high=mean+std
    p1, = ax_handle.semilogx(x_plot,mean, linewidth=2)
    ax_handle.fill_between(x_plot,low,high,alpha=0.3)
    return p1

def plot_shaded(x_plot, mean,std,ax_handle):
    # iter=len(mean)
    mean=np.array(mean)
    std=np.array(std)
    low=mean-std
    high=mean+std
    p1, = ax_handle.plot(x_plot,mean, linewidth=2, marker='o')
    ax_handle.fill_between(x_plot,low,high,alpha=0.3)
    return p1

def plot_threeline_shaded_logx(x_plot,data1_mean,data1_std,data2_mean,data2_std,data3_mean,data3_std,xlabel,ylabel,legend=["ZO-Min-Max","FO-Min-Max","FO-NoPoison"],loc='upper left',filename=None):
    # plt.figure()
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded_logx(x_plot,data1_mean,data1_std,ax_handle)
    p2=plot_shaded_logx(x_plot,data2_mean,data2_std,ax_handle)
    p3=plot_shaded_logx(x_plot,data3_mean,data3_std,ax_handle)
    ax_handle.legend([p1, p2, p3], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)
        # plt.savefig(filename)
    # plt.show()
    # plt.close()


def plot_twoline_shaded_logx(x_plot, data1_mean,data1_std,data2_mean,data2_std,xlabel,ylabel,legend,loc='upper left',filename=None):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1=plot_shaded_logx(x_plot, data1_mean,data1_std,ax_handle)
    p2=plot_shaded_logx(x_plot, data2_mean,data2_std,ax_handle)
    ax_handle.legend([p1, p2], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)
        # my_path = os.path.abspath(__file__)
        # plt.savefig('/Results_figures/' + filename)
    # plt.show()
    # plt.close()


def plot_twoline_shaded(x_plot, data1_mean, data1_std, data2_mean, data2_std, xlabel, ylabel, legend,
                             loc='upper left', filename=None):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1 = plot_shaded(x_plot, data1_mean, data1_std, ax_handle)
    p2 = plot_shaded(x_plot, data2_mean, data2_std, ax_handle)
    ax_handle.legend([p1, p2], legend, loc=loc)
    ax_handle.set_xlabel(xlabel)
    ax_handle.set_ylabel(ylabel)
    if filename != None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)
        # plt.savefig(filename)
    # plt.show()
    # plt.close()


########### fixing lambda, multiple trials plot
def multiplot_all_logx(train_poison_data, train_clean_data,test_data,lambda_x,alpha,beta,q_vec, times,epsilon, idx_coord_plot, Data_path, filename_temp):

    #### multi-random trials & multi-q
    num_q = len(q_vec)

    atrl_AG=np.zeros((len(idx_coord_plot),times,num_q))
    atrc_AG=np.zeros((len(idx_coord_plot),times,num_q))
    atec_AG=np.zeros((len(idx_coord_plot),times,num_q))
    sc_AG=np.zeros((len(idx_coord_plot),times,num_q))

    atrl_FO=np.zeros((len(idx_coord_plot),times))
    atrc_FO=np.zeros((len(idx_coord_plot),times))
    atec_FO=np.zeros((len(idx_coord_plot),times))
    sc_FO=np.zeros((len(idx_coord_plot),times))

    atrl_BL=np.zeros((len(idx_coord_plot),times))
    atrc_BL=np.zeros((len(idx_coord_plot),times))
    atec_BL=np.zeros((len(idx_coord_plot),times))

    # atrl_T_AG=np.zeros((times,num_q))
    # # atel_AG=[]
    # atrc_T_AG=np.zeros((times,num_q))
    # atec_T_AG=np.zeros((times,num_q))
    # sc_T_AG=np.zeros((times,num_q))
    # D=np.shape(train_clean_data[0])[0]-1

   #### multiple trials and multiple random direction vectors
    for i in range(0,times):
        print("Evaluation for trial %d" % i)
        for iq in range(0,len(q_vec)):
            q_rand = q_vec[iq]
            filename = filename_temp +"_q_" + str(q_rand) + "_exp_" + str(i)  ### "lambda_"+str(lambda_x[i])+"_exp_"+str(j
            AG =np.load(Data_path + "/" + "ZOAG_4_2_SL_"+filename+".npz")
            AG_iter_res=AG['AG_iter_res'][idx_coord_plot,:] ### only evaluate the points of interests

            ###### do specific evaluations for this specific q
            all_train_loss_AG = all_loss(AG_iter_res, lambda_x, train_poison_data, train_clean_data, test_data) ## F(x^t, \delta^t)
            stat_con_AG = stationary_condition(AG_iter_res, train_poison_data, train_clean_data, lambda_x=lambda_x,
                                                   alpha=alpha, beta=beta, epsilon=epsilon)
            ###### retrain under best poisoned vector
            all_train_accuracy_AG, all_test_accuracy_AG = all_acc_retrain(AG_iter_res, train_poison_data,
                                                                      train_clean_data,test_data, lambda_x)
            ### ZO-min-max
            atrl_AG[:,i,iq] = all_train_loss_AG
            atrc_AG[:,i,iq] = all_train_accuracy_AG
            atec_AG[:,i,iq] = all_test_accuracy_AG
            sc_AG[:,i,iq] = stat_con_AG


        ### first-order case
        filename = filename_temp + "_exp_" + str(i)

        FO=np.load(Data_path + "/" + "FOAG_4_2_SL_"+filename+".npz")
        FO_iter_res=FO['FO_iter_res'][idx_coord_plot,:]
        # FO_time=FO['FO_time'][idx_coord_plot]
        all_train_loss_FO=all_loss(FO_iter_res,lambda_x,train_poison_data, train_clean_data,test_data)
        all_train_accuracy_FO,all_test_accuracy_FO=all_acc_retrain(FO_iter_res,train_poison_data, train_clean_data,test_data,lambda_x)
        stat_con_FO=stationary_condition(FO_iter_res,train_poison_data, train_clean_data,lambda_x=lambda_x,alpha=alpha,beta=beta,epsilon=epsilon)
        atrl_FO[:, i] = all_train_loss_FO
        atrc_FO[:, i] = all_train_accuracy_FO
        atec_FO[:, i] = all_test_accuracy_FO
        sc_FO[:, i] = stat_con_FO

        # atrl_FO.append(all_train_loss_FO)
        # atrc_FO.append(all_train_accuracy_FO)
        # atec_FO.append(all_test_accuracy_FO)
        # sc_FO.append(stat_con_FO)

        BL=np.load(Data_path + "/" + "FOnonAdv_4_2_SL_"+filename+".npz")
        BL_iter_res=BL['FO_nonAdv_iter_res'][idx_coord_plot,:]
        BL_time=BL['FO_nonAdv_time'][idx_coord_plot]
        all_train_loss_BL =all_loss(BL_iter_res,lambda_x,train_poison_data, train_clean_data,test_data)
        all_train_accuracy_BL,all_test_accuracy_BL=all_acc(BL_iter_res,train_poison_data, train_clean_data,test_data)
        # stat_con_BL=stationary_condition(BL_iter_res,train_poison_data, train_clean_data,test_data,lambda_x=lambda_x,alpha=alpha,beta=beta,epsilon=epsilon)
        # atrl_BL.append(all_train_loss_BL)
        # atrc_BL.append(all_train_accuracy_BL)
        # atec_BL.append(all_test_accuracy_BL)
        # sc_BL.append(stat_con_BL)
        atrl_BL[:, i] = all_train_loss_BL
        atrc_BL[:, i] = all_train_accuracy_BL
        atec_BL[:, i] = all_test_accuracy_BL
        print("Ending Evaluation for trial %d" % i)


    # atrl_AG_mean,atrl_AG_std=mean_std(atrl_AG)
    # atrc_AG_mean,atrc_AG_std=mean_std(atrc_AG)
    # atec_AG_mean,atec_AG_std=mean_std(atec_AG)
    # sc_AG_mean,sc_AG_std=mean_std(sc_AG)
    #
    # atrl_FO_mean,atrl_FO_std=mean_std(atrl_FO)
    # atrc_FO_mean,atrc_FO_std=mean_std(atrc_FO)
    # atec_FO_mean,atec_FO_std=mean_std(atec_FO)
    # sc_FO_mean,sc_FO_std=mean_std(sc_FO)
    #
    # atrl_BL_mean,atrl_BL_std=mean_std(atrl_BL)
    # atrc_BL_mean,atrc_BL_std=mean_std(atrc_BL)
    # atec_BL_mean,atec_BL_std=mean_std(atec_BL)

    # filename_plot = "lambda_" + str(lambda_x) + "_exp"
    filename_plot = "lambda_" + str(int(lambda_x*1000)) + "_exp"

    #### objective value versus iterations
    for iq in range(0, len(q_vec)):
        filename_plot_q = "lambda_" + str(int(lambda_x*1000)) + "_q_" + str(q_vec[iq]) + "_exp"

        plot_threeline_shaded_logx(np.array(idx_coord_plot)+1, np.mean(atrl_AG[:,:,iq],axis=1), np.std(atrl_AG[:,:,iq],axis=1),
                                   np.mean(atrl_FO, axis=1), np.std(atrl_FO, axis=1),
                                   np.mean(atrl_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)),0*np.ones(len(idx_coord_plot)),
                             "Number of iterations", "Objective value", legend=["ZO-Min-Max", "FO-Min-Max", "No Poison"],
                             loc='best', filename="train_loss_shaded_SL" + filename_plot_q + ".pdf")

        plot_threeline_shaded_logx(np.array(idx_coord_plot)+1, np.mean(atrc_AG[:,:,iq],axis=1), np.std(atrc_AG[:,:,iq],axis=1),
                                   np.mean(atrc_FO, axis=1), np.std(atrc_FO, axis=1),
                                   np.mean(atrc_BL, axis=1)[-1] * np.ones(len(idx_coord_plot)),0*np.ones(len(idx_coord_plot)),
                                 "Number of iterations", "Training accuracy", legend=["ZO-Min-Max", "FO-Min-Max" , "No Poison"],
                                 loc='best', filename="train_accuracy_shaded_SL" + filename_plot_q + ".pdf")

        plot_threeline_shaded_logx(np.array(idx_coord_plot)+1, np.mean(atec_AG[:,:,iq],axis=1), np.std(atec_AG[:,:,iq],axis=1),
                                   np.mean(atec_FO, axis=1), np.std(atec_FO, axis=1),
                                   np.mean(atec_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)), 0*np.ones(len(idx_coord_plot)),
                                 "Number of iterations", "Testing accuracy", legend=["ZO-Min-Max", "FO-Min-Max" , "No Poison"],
                                 loc='best', filename="test_accuracy_shaded_SL" + filename_plot_q + ".pdf")

        plot_twoline_shaded_logx(np.array(idx_coord_plot)+1,  np.mean(sc_AG[:,:,iq],axis=1), np.std(sc_AG[:,:,iq],axis=1),
                                 np.mean(sc_FO, axis=1), np.std(sc_FO, axis=1),
                       "Number of iterations","Stationary condition",legend=["ZO-Min-Max","FO-Min-Max"],loc='upper left',
                                 filename="stationary_condition_shaded_SL"+filename_plot_q+".pdf")

    plot_twoline_shaded(q_vec, np.mean(sc_AG[-1,:,:],axis=0), np.std(sc_AG[-1,:,:],axis=0),
                        np.mean(sc_FO[-1,:]) * np.ones(len(q_vec)),
                        np.std(sc_FO[-1,:])* np.ones(len(q_vec)),
                    "Number of random directions q", "Stationary condition", legend=["ZO-Min-Max", "FO-Min-Max"],
                    loc='best', filename="stationary_condition_allQ" + filename_plot + ".pdf")


def multiplot_all_logx_updated(train_poison_data, train_clean_data,test_data,lambda_x,alpha,beta,q_vec, times,epsilon, idx_coord_plot, Data_path, filename_temp):

    #### multi-random trials & multi-q
    num_q = len(q_vec)

    atrl_AG=np.zeros((len(idx_coord_plot),times,num_q))
    atrc_AG=np.zeros((len(idx_coord_plot),times,num_q))
    atec_AG=np.zeros((len(idx_coord_plot),times,num_q))
    sc_AG=np.zeros((len(idx_coord_plot),times,num_q))

    atrl_FO=np.zeros((len(idx_coord_plot),times))
    atrc_FO=np.zeros((len(idx_coord_plot),times))
    atec_FO=np.zeros((len(idx_coord_plot),times))
    sc_FO=np.zeros((len(idx_coord_plot),times))

    atrl_BL=np.zeros((len(idx_coord_plot),times))
    atrc_BL=np.zeros((len(idx_coord_plot),times))
    atec_BL=np.zeros((len(idx_coord_plot),times))

    # atrl_T_AG=np.zeros((times,num_q))
    # # atel_AG=[]
    # atrc_T_AG=np.zeros((times,num_q))
    # atec_T_AG=np.zeros((times,num_q))
    # sc_T_AG=np.zeros((times,num_q))
    # D=np.shape(train_clean_data[0])[0]-1

   #### multiple trials and multiple random direction vectors
    for i in range(0,times):
        print("Evaluation for trial %d" % i)
        for iq in range(0,len(q_vec)):
            q_rand = q_vec[iq]
            filename = filename_temp +"_q_" + str(q_rand) + "_exp_" + str(i)  ### "lambda_"+str(lambda_x[i])+"_exp_"+str(j
            AG =np.load(Data_path + "/" + "ZOAG_4_2_SL_"+filename+".npz")
            AG_iter_res=AG['AG_iter_res'][idx_coord_plot,:] ### only evaluate the points of interests

            ###### do specific evaluations for this specific q
            all_train_loss_AG = all_loss(AG_iter_res, lambda_x, train_poison_data, train_clean_data, test_data) ## F(x^t, \delta^t)
            stat_con_AG = stationary_condition(AG_iter_res, train_poison_data, train_clean_data, lambda_x=lambda_x,
                                                   alpha=alpha, beta=beta, epsilon=epsilon)
            ###### retrain under best poisoned vector
            all_train_accuracy_AG, all_test_accuracy_AG = all_acc_retrain(AG_iter_res, train_poison_data,
                                                                      train_clean_data,test_data, lambda_x)
            ### ZO-min-max
            atrl_AG[:,i,iq] = all_train_loss_AG
            atrc_AG[:,i,iq] = all_train_accuracy_AG
            atec_AG[:,i,iq] = all_test_accuracy_AG
            sc_AG[:,i,iq] = stat_con_AG


        ### first-order case
        filename = filename_temp + "_exp_" + str(i)

        FO=np.load(Data_path + "/" + "FOAG_4_2_SL_"+filename+".npz")
        FO_iter_res=FO['FO_iter_res'][idx_coord_plot,:]
        # FO_time=FO['FO_time'][idx_coord_plot]
        all_train_loss_FO=all_loss(FO_iter_res,lambda_x,train_poison_data, train_clean_data,test_data)
        all_train_accuracy_FO,all_test_accuracy_FO=all_acc_retrain(FO_iter_res,train_poison_data, train_clean_data,test_data,lambda_x)
        stat_con_FO=stationary_condition(FO_iter_res,train_poison_data, train_clean_data,lambda_x=lambda_x,alpha=alpha,beta=beta,epsilon=epsilon)
        atrl_FO[:, i] = all_train_loss_FO
        atrc_FO[:, i] = all_train_accuracy_FO
        atec_FO[:, i] = all_test_accuracy_FO
        sc_FO[:, i] = stat_con_FO


        BL=np.load(Data_path + "/" + "FOnonAdv_4_2_SL_"+filename+".npz")
        BL_iter_res=BL['FO_nonAdv_iter_res'][idx_coord_plot,:]
        BL_time=BL['FO_nonAdv_time'][idx_coord_plot]
        all_train_loss_BL =all_loss(BL_iter_res,lambda_x,train_poison_data, train_clean_data,test_data)
        all_train_accuracy_BL,all_test_accuracy_BL=all_acc(BL_iter_res,train_poison_data, train_clean_data,test_data)

        atrl_BL[:, i] = all_train_loss_BL
        atrc_BL[:, i] = all_train_accuracy_BL
        atec_BL[:, i] = all_test_accuracy_BL
        print("Ending Evaluation for trial %d" % i)




    # filename_plot = "lambda_" + str(lambda_x) + "_exp"
    filename_plot = "lambda_" + str(int(lambda_x*1000)) + "_exp"

    #### objective value versus iterations
    for iq in range(0, len(q_vec)):
        filename_plot_q = "lambda_" + str(int(lambda_x*1000)) + "_q_" + str(q_vec[iq]) + "_exp"

        plot_threeline_shaded_logx(np.array(idx_coord_plot)+1, np.mean(atrl_AG[:,:,iq],axis=1), np.std(atrl_AG[:,:,iq],axis=1),
                                   np.mean(atrl_FO, axis=1), np.std(atrl_FO, axis=1),
                                   np.mean(atrl_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)),0*np.ones(len(idx_coord_plot)),
                             "Number of iterations", "Objective value", legend=["ZO-Min-Max", "FO-Min-Max", "No Poison"],
                             loc='best', filename="train_loss_shaded_SL" + filename_plot_q + ".pdf")

        plot_threeline_shaded_logx(np.array(idx_coord_plot)+1, np.mean(atrc_AG[:,:,iq],axis=1), np.std(atrc_AG[:,:,iq],axis=1),
                                   np.mean(atrc_FO, axis=1), np.std(atrc_FO, axis=1),
                                   np.mean(atrc_BL, axis=1)[-1] * np.ones(len(idx_coord_plot)),0*np.ones(len(idx_coord_plot)),
                                 "Number of iterations", "Training accuracy", legend=["ZO-Min-Max", "FO-Min-Max" , "No Poison"],
                                 loc='best', filename="train_accuracy_shaded_SL" + filename_plot_q + ".pdf")

        plot_threeline_shaded_logx(np.array(idx_coord_plot)+1, np.mean(atec_AG[:,:,iq],axis=1), np.std(atec_AG[:,:,iq],axis=1),
                                   np.mean(atec_FO, axis=1), np.std(atec_FO, axis=1),
                                   np.mean(atec_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)), 0*np.ones(len(idx_coord_plot)),
                                 "Number of iterations", "Testing accuracy", legend=["ZO-Min-Max", "FO-Min-Max" , "No Poison"],
                                 loc='best', filename="test_accuracy_shaded_SL" + filename_plot_q + ".pdf")

        plot_twoline_shaded_logx(np.array(idx_coord_plot)+1,  np.mean(sc_AG[:,:,iq],axis=1), np.std(sc_AG[:,:,iq],axis=1),
                                 np.mean(sc_FO, axis=1), np.std(sc_FO, axis=1),
                       "Number of iterations","Stationary gap",legend=["ZO-Min-Max","FO-Min-Max"],loc='upper left',
                                 filename="stationary_condition_shaded_SL"+filename_plot_q+".pdf")

    #### stationary condition versus q
    plot_twoline_shaded(q_vec, np.mean(sc_AG[-1,:,:],axis=0), np.std(sc_AG[-1,:,:],axis=0),
                        np.mean(sc_FO[-1,:]) * np.ones(len(q_vec)),
                        np.std(sc_FO[-1,:])* np.ones(len(q_vec)),
                    "Number of random directions q", "Stationary gap", legend=["ZO-Min-Max", "FO-Min-Max"],
                    loc='best', filename="stationary_condition_allQ" + filename_plot + ".pdf")

    #### plot stationary condition for all Qs
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    legend_txt = []
    for i in range(0,len(q_vec)):
        mean_tmp = np.mean(sc_AG[:,:,i],axis=1)
        # std_temp = np.std(sc_AG[:,:,i],axis=1)
        # low = mean_tmp - std_temp
        # high = mean_tmp + std_temp
        p1,=ax_handle.semilogx(np.array(idx_coord_plot)+1,mean_tmp,linewidth=2)
        # ax_handle.fill_between(np.array(idx_coord_plot)+1, low, high, alpha=0.3)
        legend_txt.append("ZO-Min-Max: q="+str(q_vec[i]))

    p1, = ax_handle.semilogx(np.array(idx_coord_plot) + 1,  np.mean(sc_FO, axis=1), linewidth=2)
    legend_txt.append("FO-Min-Max")

    ax_handle.legend(legend_txt, loc='best')
    ax_handle.set_xlabel("Number of iterations")
    ax_handle.set_ylabel("Stationary gap")
    my_path = os.path.join('Results_figures', "stationary_condition_muli_q.pdf")
    plt.tight_layout()
    plt.savefig(my_path)
    #plt.show()
    # plt.close()

    #### plot loss for all Qs
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    legend_txt = []
    for i in range(0,len(q_vec)):
        mean_tmp = np.mean(atrl_AG[:,:,i],axis=1)
        # std_temp = np.std(sc_AG[:,:,i],axis=1)
        # low = mean_tmp - std_temp
        # high = mean_tmp + std_temp
        p1,=ax_handle.semilogx(np.array(idx_coord_plot)+1,mean_tmp,linewidth=2)
        # ax_handle.fill_between(np.array(idx_coord_plot)+1, low, high, alpha=0.3)
        legend_txt.append("ZO-Min-Max: q="+str(q_vec[i]))

    p1, = ax_handle.semilogx(np.array(idx_coord_plot) + 1,  np.mean(atrl_FO, axis=1), linewidth=2)
    legend_txt.append("FO-Min-Max")

    p1, = ax_handle.semilogx(np.array(idx_coord_plot) + 1,  np.mean(atrl_BL, axis=1)[-1]*np.ones(len(idx_coord_plot)), linewidth=2)
    legend_txt.append("No Poison")

    ax_handle.legend(legend_txt, loc='best')
    ax_handle.set_xlabel("Number of iterations")
    ax_handle.set_ylabel("Objective value")
    my_path = os.path.join('Results_figures', "Objective_value_muli_q.pdf")
    plt.tight_layout()
    plt.savefig(my_path)



########################################################################################
######################## Main function to rn all algorithms and plots #############################
########################################################################################
if __name__=="__main__":
    n_tr = 1000
    train_ratio = 0.7
    poison_ratio = 0.1 ### 0.05
    D_x = 100
    x_gt = 1 * np.ones(D_x)

    sigma2 = 1
    noise_std = 1e-3

    batch_sz = 100
    n_iters = 50000

    eps_perturb = 2
    lr_x = 0.05
    lr_delta = 0.02 ### 0.02

    ############## Part I: data generation   ######################
    flag_dataGeneration = 1
    flag_dataLoad = 1
    Data_filename = "D_4_2_SL"
    Data_path = os.path.dirname(os.path.realpath(__file__))+"/data_4_2"
    if not os.path.exists(Data_path):
        os.makedirs(Data_path)
    if not os.path.exists(os.path.dirname(os.path.realpath(__file__))+"/Results_figures"):
        os.makedirs(os.path.dirname(os.path.realpath(__file__))+"/Results_figures")
    if flag_dataGeneration:
        generate_data(n_tr, sigma2, x_gt, noise_std, Data_path, Data_filename)
        train_data, train_poison_data, train_clean_data, test_data = generate_train_and_test_data(train_ratio, poison_ratio, Data_path,  Data_filename, True)
        index = generate_index(np.size(train_clean_data,0), batch_sz, n_iters, Data_path, Data_filename)
        index_batch = generate_index(np.size(train_clean_data,0), int(round(1*np.size(train_clean_data,0))),
                                     n_iters, Data_path, Data_filename+"batch")

    else:
        if flag_dataLoad:
            train_data, train_poison_data, train_clean_data, test_data = load_train_and_test_data(poison_ratio,Data_path,Data_filename)
            index = load_index(Data_path, Data_filename)
            index_batch = load_index(Data_path, Data_filename+"batch")

    # ### full batch
    # index = index_batch.copy()
    ############## Part II: poisoning attack learning for different q, lambda, multiple trials
    flag_train = 1
    lambda_x =  [1e-3]  #[1e-3, 1e-2]
    q_vec = [1,5,10,20] #30 # ### multiple random direction vectors
    n_trials = 10 #3



    if flag_train:
        for i in range(0,len(lambda_x)):
            for j in range(0,n_trials):
                #### if attack loss is large, then the classifier is accurate and no adv.
                # x_ini = x_gt + 1*np.random.normal(0, 1, D_x) ### initial x from a point close to ground truth (no adv. is considered)
                x_ini_tmp = np.random.normal(0, 1, D_x)
                delta_ini_tmp = np.zeros(D_x) #np.random.normal(0, 0.01, D_x)  #np.random.normal(0, 0.1, D_x)  # np.zeros(D_x)  # np.random.uniform(-eps_perturb,eps_perturb,D_x) ### inital delta

                #### stationary condition for initial point
                thr_stat = 0
                for i_test in range(0,100):
                    iter_res = np.zeros((1,D_x+D_x))
                    iter_res[0,:D_x] = delta_ini_tmp
                    iter_res[0,D_x:] = x_ini_tmp
                    stat_temp = stationary_condition(iter_res, train_poison_data, train_clean_data, lambda_x[i], lr_delta, lr_x, eps_perturb)
                    if stat_temp > thr_stat:
                        thr_stat = stat_temp
                        x_ini = x_ini_tmp
                        delta_ini = delta_ini_tmp
                        print("Stationary condition = %4.3f" % stat_temp)
                    x_ini_tmp =  np.random.normal(0, 1, D_x) # np.random.uniform(-2, 2, D_x)
                    delta_ini_tmp = np.zeros(D_x) #np.random.uniform(-1, 1, D_x)  # np.random.normal(0, 0.01, D_x)  #np.random.normal(0, 0.1, D_x)  # np.zeros(D_x)  # np.random.uniform(-eps_perturb,eps_perturb,D_x) ### inital delta



                #### ZO-Min-Max
                #### AG_sol_track: first D_x dimension is x, and the last D_x dimension is delta
                for iq in range(0,len(q_vec)):
                    q_rand = q_vec[iq]
                    x_opt_AG, AG_sol_track, AG_time_track = AG_main_batch_SL(train_poison_data, train_clean_data,
                                    x_ini, delta_ini, eps_perturb, n_iters, x_gt, index, lambda_x[i], lr_delta, lr_x, q_rand,
                                     Data_path, filename="lambda_"+str(lambda_x[i])+"_q_"+str(q_rand)+"_exp_"+str(j))


                # var_results_track_AG[i,j,:,:] = AG_sol_track

                #### FO-Min-Max
                x_opt_FO, FO_sol_track, FO_time_track \
                    = FO_main_batch_SL(train_poison_data, train_clean_data, x_ini, delta_ini, eps_perturb, n_iters, x_gt,
                                       index, lambda_x[i], lr_delta, lr_x,  Data_path,
                                       filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j))
                ### index_batch
                # var_results_track_FO[i, j, :, :] = FO_sol_track

                #### No-adv effect
                x_opt_FO_nonAdv, FO_nonAdv_sol_track, FO_nonAdv_time_track = FO_nonAdv_main_batch_SL(train_poison_data, train_clean_data, x_ini,
                                                                                                     delta_ini, eps_perturb, n_iters, x_gt,
                                                                                                     index, lambda_x[i], lr_delta, lr_x, Data_path,
                                                                                                     filename="lambda_" + str(lambda_x[i]) + "_exp_" + str(j))

                # var_results_track_FO_nonAdv[i, j, :, :] = FO_nonAdv_sol_track


    ################################# plot figures for part II
    flag_plot = 1
    if flag_plot:
        idx_coordinate_plot = range(0,n_iters,50)
        for i in range(0, len(lambda_x)):
            filename_temp = "lambda_" + str(lambda_x[i])
            multiplot_all_logx_updated(train_poison_data, train_clean_data, test_data, lambda_x[i], lr_delta,lr_x, q_vec,  n_trials, eps_perturb,idx_coordinate_plot,Data_path, filename_temp)


    ############## Part III: poisoning attack learning versus different poisoning ratios
    flag_poison_ratio_test = 1
    flag_dataGeneration_multiple = 1
    # flag_dataLoad_multiple = 0

    flag_plot_poisonRatio = 1

    lambda_x = 1e-3
    filename_plot = "lambda_" + str(lambda_x) + "PoisonRatio_exp"
    poison_ratio_vec = np.array([0.00, 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2])



    n_trials = 5 ##3

    if flag_poison_ratio_test:
        trAcc_ZOAG = np.zeros(( len(poison_ratio_vec),n_trials))
        teAcc_ZOAG = np.zeros((len(poison_ratio_vec), n_trials))
        trAcc_FOAG = np.zeros(( len(poison_ratio_vec),n_trials))
        teAcc_FOAG = np.zeros((len(poison_ratio_vec), n_trials))

        Data_filename = "D_4_2_Common_" + str(lambda_x) + "MultipleRatio_SL"
        generate_data(n_tr, sigma2, x_gt, noise_std, Data_path, Data_filename) ### generate whole dataset
        q_rand = 10 ### number of random direction vectors
        n_retrain = 1000 ### retrain steps


        ############## Generate Dataset: Whole dataset has been generated, and batch sets have been generated ####################
        if flag_dataGeneration_multiple:
            for ii in range(0, len(poison_ratio_vec)):
                poison_ratio = poison_ratio_vec[ii]
                train_data, train_poison_data, train_clean_data, test_data = generate_train_and_test_data(train_ratio,
                                                                                                          poison_ratio,
                                                                                                          Data_path,
                                                                                                          Data_filename,
                                                                                                          False)
                # index = generate_index(np.size(train_clean_data, 0), batch_sz, n_iters, Data_path, Data_filename)
                # index_batch = generate_index(np.size(train_clean_data, 0), int(round(0.8*np.size(train_clean_data,0))),
                #                                  n_iters, Data_path,Data_filename + "batch")

        ############# Training algorithm
        for ii in range(0, len(poison_ratio_vec)):
            poison_ratio = poison_ratio_vec[ii]
            ### generate data under each poison ratio
            train_data, train_poison_data, train_clean_data, test_data = load_train_and_test_data(poison_ratio,
                                                                                                  Data_path,
                                                                                                  Data_filename)

            index = generate_index(np.size(train_clean_data, 0), batch_sz, n_iters, Data_path, Data_filename) ### offline batch sets

            # if flag_dataGeneration_multiple:
            #     # generate_data(n_tr, sigma2, x_gt, noise_std, Data_filename)
            #     train_data, train_poison_data, train_clean_data, test_data = generate_train_and_test_data(train_ratio,
            #                                                                                               poison_ratio,
            #                                                                                               Data_path,
            #                                                                                               Data_filename,
            #                                                                                               False)
            #     ### batch_sz is only for clean data
            #     index = generate_index(np.size(train_clean_data, 0), batch_sz, n_iters, Data_path, Data_filename)
            #     index_batch = generate_index(np.size(train_clean_data, 0), int(round(0.8*np.size(train_clean_data,0))),
            #                                  n_iters, Data_path,
            #                                  Data_filename + "batch")
            #
            # else:
            #     if flag_dataLoad_multiple:
            #         train_data, train_poison_data, train_clean_data, test_data = load_train_and_test_data(poison_ratio, Data_path, Data_filename)
            #         index = load_index(Data_path, Data_filename)
            #         index_batch = load_index(Data_path, Data_filename + "batch")

            # # ### full batch
            # index = index_batch.copy()

            for j in range(0, n_trials):
                    #### if attack loss is large, then the classifier is accurate and no adv.
                    # x_ini = x_gt + 1 * np.random.normal(0, 0.01,
                    #                                     D_x)  ### initial x from a point close to ground truth (no adv. is considered)
                    x_ini_tmp = np.random.normal(0, 1, D_x)
                    delta_ini_tmp = np.zeros(D_x)  # np.random.normal(0, 0.01, D_x)  #np.random.normal(0, 0.1, D_x)  # np.zeros(D_x)  # np.random.uniform(-eps_perturb,eps_perturb,D_x) ### inital delta

                    #### stationary condition for initial point
                    thr_stat = 0
                    for i_test in range(0, 100):
                        iter_res = np.zeros((1, D_x + D_x))
                        iter_res[0, :D_x] = delta_ini_tmp
                        iter_res[0, D_x:] = x_ini_tmp
                        stat_temp = stationary_condition(iter_res, train_poison_data, train_clean_data, lambda_x,
                                                         lr_delta, lr_x, eps_perturb)
                        if stat_temp > thr_stat:
                            thr_stat = stat_temp
                            x_ini = x_ini_tmp
                            delta_ini = delta_ini_tmp
                            print("Stationary condition = %4.3f" % stat_temp)
                        x_ini_tmp = np.random.normal(0, 1, D_x)  # np.random.uniform(-2, 2, D_x)
                        delta_ini_tmp = np.zeros(D_x)  # np.random.uniform(-1, 1, D_x)  # np.random.normal(0, 0.01, D_x)  #np.random.normal(0, 0.1, D_x)  # np.zeros(D_x)  # np.random.uniform(-eps_perturb,eps_perturb,D_x) ### inital delta

                    #### ZO-Min-Max
                    #### AG_sol_track: first D_x dimension is x, and the last D_x dimension is delta
                    x_opt_AG, AG_sol_track, AG_time_track = AG_main_batch_SL(train_poison_data, train_clean_data,
                                                                             x_ini, delta_ini, eps_perturb, n_iters,
                                                                             x_gt, index, lambda_x, lr_delta, lr_x, q_rand,
                                                                             Data_path, filename="posRatio" + str(poison_ratio) +
                                                                                      "lambda_" + str(lambda_x) +
                                                                                      "_q_" + str(q_rand) +
                                                                                      "_exp_" + str(j))
                    ### if posratio = 0, then grad_ZO over detla

                    ### testing error / training error
                    x_retrain = FO_retrain_poison(train_poison_data, train_clean_data, AG_sol_track[-1,D_x:2*D_x],
                                                   AG_sol_track[-1, 0:D_x], n_retrain, lambda_x, 0.05)
                    teAcc_ZOAG[ii,j] = acc_for_D(x_retrain, test_data)
                    trAcc_ZOAG[ii,j] = acc_for_PoisonD(x_retrain, AG_sol_track[-1,0:D_x], train_poison_data,
                                                       train_clean_data)

                    #### FO-Min-Max
                    x_opt_FO, FO_sol_track, FO_time_track \
                        = FO_main_batch_SL(train_poison_data, train_clean_data, x_ini, delta_ini, eps_perturb, n_iters,
                                           x_gt, index, lambda_x, lr_delta, lr_x, Data_path,
                                           filename="posRatio" + str(poison_ratio) +
                                                    "lambda_" + str(lambda_x) + "_exp_" + str(j))


                    ### testing error / training error
                    x_retrain_FO = FO_retrain_poison(train_poison_data, train_clean_data, FO_sol_track[-1,D_x:2*D_x],
                                                  FO_sol_track[-1, 0:D_x], n_retrain, lambda_x, 0.05)
                    teAcc_FOAG[ii,j] = acc_for_D(x_retrain_FO, test_data)
                    trAcc_FOAG[ii,j] = acc_for_PoisonD(x_retrain_FO, FO_sol_track[-1,0:D_x], train_poison_data,
                                                       train_clean_data)



        np.savez(Data_path + "/" +"MultiRatio_AG_4_2_SL_"+filename_plot+".npz",trAcc_ZOAG=trAcc_ZOAG,
                 teAcc_ZOAG=teAcc_ZOAG,trAcc_FOAG=trAcc_FOAG,teAcc_FOAG = teAcc_FOAG)

    if flag_plot_poisonRatio:

            #### load results
            results_tmp = np.load(Data_path + "/" + "MultiRatio_AG_4_2_SL_"+filename_plot+".npz")
            trAcc_ZOAG = results_tmp['trAcc_ZOAG']
            teAcc_ZOAG = results_tmp['teAcc_ZOAG']
            trAcc_FOAG = results_tmp['trAcc_FOAG']
            teAcc_FOAG = results_tmp['teAcc_FOAG']

            idx_plt = np.arange(0,len(poison_ratio_vec)).astype(int)

            plot_twoline_shaded(poison_ratio_vec[idx_plt]*100, np.mean(trAcc_ZOAG[idx_plt,:],axis=1), np.std(trAcc_ZOAG[idx_plt,:],axis=1),
                                np.mean(trAcc_FOAG[idx_plt,:],axis=1), np.std(trAcc_FOAG[idx_plt,:],axis=1),
                                     "Poisoning ratio (%)", "Training accuracy", legend=["ZO-Min-Max", "FO-Min-Max"],
                                     loc='best', filename="train_accuracy_vs_posRatio" + filename_plot + ".pdf")

            idx_sort= np.argsort(-np.mean(trAcc_ZOAG[idx_plt,:],axis=1))
            idx_sort2 = np.argsort(-np.mean(trAcc_FOAG[idx_plt,:],axis=1))
            plot_twoline_shaded(poison_ratio_vec[idx_plt]*100, np.mean(trAcc_ZOAG[idx_plt,:],axis=1)[idx_sort], np.std(trAcc_ZOAG[idx_plt,:],axis=1)[idx_sort],
                                np.mean(trAcc_FOAG[idx_plt,:],axis=1)[idx_sort2], np.std(trAcc_FOAG[idx_plt,:],axis=1)[idx_sort2],
                                     "Poisoning ratio (%)", "Training accuracy", legend=["ZO-Min-Max", "FO-Min-Max"],
                                     loc='best', filename="train_accuracy_Sort_vs_posRatio" + filename_plot + ".pdf")


            plot_twoline_shaded(poison_ratio_vec[idx_plt]*100, np.mean(teAcc_ZOAG[idx_plt,:],axis=1), np.std(teAcc_ZOAG[idx_plt,:],axis=1),
                                np.mean(teAcc_FOAG[idx_plt,:],axis=1), np.std(teAcc_FOAG[idx_plt,:],axis=1),
                                     "Poisoning ratio (%)", "Testing accuracy", legend=["ZO-Min-Max", "FO-Min-Max"],
                                     loc='best', filename="test_accuracy_vs_posRatio" + filename_plot + ".pdf")

            idx_sort= np.argsort(-np.mean(teAcc_ZOAG[idx_plt,:],axis=1))
            idx_sort2 = np.argsort(-np.mean(teAcc_FOAG[idx_plt,:],axis=1))
            plot_twoline_shaded(poison_ratio_vec[idx_plt]*100, np.mean(teAcc_ZOAG[idx_plt,:],axis=1)[idx_sort], np.std(teAcc_ZOAG[idx_plt,:],axis=1)[idx_sort],
                                np.mean(teAcc_FOAG[idx_plt,:],axis=1)[idx_sort2], np.std(teAcc_FOAG[idx_plt,:],axis=1)[idx_sort2],
                                     "Poisoning ratio (%)", "Testing accuracy", legend=["ZO-Min-Max", "FO-Min-Max"],
                                     loc='best', filename="test_accuracy_Sort_vs_posRatio" + filename_plot + ".pdf")
