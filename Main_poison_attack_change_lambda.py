from __future__ import print_function
import numpy as np
import random
import time
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 16})
plt.switch_backend('agg')  ### running for CCC
import os
import pdb

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

def generate_poison_data_K_fold(K, poison_ratio, Data_path, filename, shuffle=False):#load generated data, get poison data
    data=np.load(Data_path + "/" + filename + ".npz")
    data=data['data']
    if shuffle:
        shuffled_indexes=list(range(0,n))
        random.shuffle(shuffled_indexes)
        data=data[shuffled_indexes]
    full_data = []
    n=np.shape(data)[0]
    train_data = []
    train_poison_data = []
    val_data = []
    train_clean_data = []
    for i in range(0, K):
        index = list(range(0, n))
        val_index = list(range(round(i*n/K),round((i+1)*n/K)))
        train_index = [index[i] for i in range(len(index)) if (i not in val_index)]
        train_data.append(data[train_index])
        val_data.append(data[val_index])
        train_poison_data.append((train_data[-1])[range(0,int(np.round(len(train_index)*poison_ratio)))])
        train_clean_data.append((train_data[-1])[range(int(np.round(len(train_index)*poison_ratio)),len(train_index))])
    np.savez(Data_path + "/" + "data_4_2_SL_" +"PosRatio_" + str(poison_ratio) + "_" + filename + ".npz",train_data = train_data, train_clean_data = train_clean_data, train_poison_data = train_poison_data, val_data = val_data)
    return train_data,train_poison_data,val_data

def load_train_and_val_data(poison_ratio, Data_path, filename):
    data = np.load(Data_path + "/" + "data_4_2_SL_" +"PosRatio_" + str(poison_ratio) + "_" + filename + ".npz")
    return data['train_data'],data['train_clean_data'],data['train_poison_data'],data['val_data']

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
    ### func: loss function
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

        if i%1000 == 0:
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
        if i%1000 == 0:
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
        if i%1000 == 0:
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
        if i%1000 == 0:
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

def plot_lambda(lambda_x,acc,ylabel,legend,filename,count=-1):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    lambda_x_new = []
    if count>0:
        for i in range(0, count):
            lambda_x_new.append(lambda_x[i])
        p1, = ax_handle.semilogx(lambda_x_new,acc[0][0:count],'*', linewidth=2, linestyle="-")
        p2, = ax_handle.semilogx(lambda_x_new,acc[1][0:count],'*', linewidth=2, linestyle="-")
        p3, = ax_handle.semilogx(lambda_x_new,acc[2][0:count],'*', linewidth=2, linestyle="-")
    else:
        p1, = ax_handle.semilogx(lambda_x,acc[0],'*', linewidth=2, linestyle="-")
        p2, = ax_handle.semilogx(lambda_x,acc[1],'*', linewidth=2, linestyle="-")
        p3, = ax_handle.semilogx(lambda_x,acc[2],'*', linewidth=2, linestyle="-")
    ax_handle.legend([p1, p2, p3], legend, loc='best')
    ax_handle.set_xlabel('Lambda')
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)

def plot_lambda_one90line(lambda_x,acc,ylabel,filename):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1, = ax_handle.semilogx(lambda_x,acc, '*', linewidth=2, linestyle="-")
    p2, = ax_handle.semilogx(lambda_x,0.9*np.ones(len(lambda_x)), linewidth=2, linestyle="--")
    ax_handle.set_xlabel('Lambda')
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)

def plot_lambda_90line(lambda_x,acc,ylabel,legend,filename):
    fig_handle = plt.figure()
    ax_handle = fig_handle.add_subplot(1, 1, 1)
    p1, = ax_handle.semilogx(lambda_x,acc[0], '*', linewidth=2, linestyle="-")
    p2, = ax_handle.semilogx(lambda_x,acc[1], '*', linewidth=2, linestyle="-")
    p3, = ax_handle.semilogx(lambda_x,acc[2], '*', linewidth=2, linestyle="-")
    p4, = ax_handle.semilogx(lambda_x,0.9*np.ones(len(acc[0])), linewidth=2, linestyle="--")
    ax_handle.legend([p1, p2, p3], legend, loc='best')
    ax_handle.set_xlabel('Lambda')
    ax_handle.set_ylabel(ylabel)
    if filename!=None:
        plt.tight_layout()
        my_path = os.path.join('Results_figures',filename)
        plt.savefig(my_path)

########################################################################################
######################## Main function to rn all algorithms and plots #############################
########################################################################################
if __name__ == "__main__":
    run_flag = 1
    lambda_x_list = [1e-5,1e-4,1e-3,1e-2,1e-1,1,100]
    Data_path = os.path.dirname(os.path.realpath(__file__)) + "/data_4_2"
    if run_flag:
        n_tr = 1000
    
        D_x = 100
        x_gt = 1 * np.ones(D_x)

        sigma2 = 1
        noise_std = 1e-3

        batch_sz = 100

        n_iters = 50000
        n_retrain = 5000

        eps_perturb = 2
        lr_x = 0.05
        lr_delta = 0.02

        flag_poison_ratio_test = 1

        #lambda_x_list = np.logspace(-5, 1, num_lambda)    
    
        num_lambda = len(lambda_x_list)

        poison_ratio_vec = np.array([0.15])

        flag_regenerate_data = 1
        flag_regenerate_index = 1
        K_fold = 5

        if flag_poison_ratio_test:
            trAcc_ZOAG = np.zeros((num_lambda, len(poison_ratio_vec), K_fold))
            teAcc_ZOAG = np.zeros((num_lambda, len(poison_ratio_vec), K_fold))
            trAcc_clean = np.zeros((num_lambda, len(poison_ratio_vec), K_fold))
            teAcc_clean = np.zeros((num_lambda, len(poison_ratio_vec), K_fold))
            trAcc_FOAG = np.zeros((num_lambda, len(poison_ratio_vec), K_fold))
            teAcc_FOAG = np.zeros((num_lambda, len(poison_ratio_vec), K_fold))
            Data_filename = "D_4_2_Common_" + "MultipleRatio_SL"
            if flag_regenerate_data:
                generate_data(n_tr, sigma2, x_gt, noise_std, Data_path, Data_filename) ### generate whole dataset
            q_rand = 10
        

            for ii in range(0, len(poison_ratio_vec)):
                poison_ratio = poison_ratio_vec[ii]
                if flag_regenerate_data:
                    generate_poison_data_K_fold(K_fold,poison_ratio,Data_path,Data_filename,False)
                train_data_all,train_clean_data_all,train_poison_data_all,val_data_all = load_train_and_val_data(poison_ratio,Data_path,Data_filename)

            

                for iii in range(0,num_lambda):
                    lambda_x = lambda_x_list[iii]
                    for j in range(0, K_fold):
                        train_data = train_data_all[j]
                        train_clean_data = train_clean_data_all[j]
                        train_poison_data = train_poison_data_all[j]
                        test_data = val_data_all[j]
                        if flag_regenerate_index:
                            index = generate_index(np.size(train_clean_data, 0), batch_sz, n_iters, Data_path, Data_filename+"_"+str(lambda_x) + "_" + str(j))
                        else:
                            index = load_index(Data_path,Data_filename+"_"+str(lambda_x) + "_" + str(j))
                        x_ini_tmp = np.random.normal(0, 1, D_x)
                        delta_ini_tmp = np.zeros(D_x)


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

                        x_opt_AG, AG_sol_track, AG_time_track = AG_main_batch_SL(train_poison_data, train_clean_data,
                                                                                x_ini, delta_ini, eps_perturb, n_iters,
                                                                                x_gt, index, lambda_x, lr_delta, lr_x, q_rand,
                                                                                Data_path, filename="posRatio" + str(poison_ratio) +
                                                                                        "lambda_" + str(lambda_x) +
                                                                                        "_q_" + str(q_rand) +
                                                                                        "_exp_" + str(j))
                        x_opt_FO, FO_sol_track, FO_time_track = FO_main_batch_SL(train_poison_data, train_clean_data, x_ini, delta_ini, eps_perturb, n_iters,
                                                                x_gt, index, lambda_x, lr_delta, lr_x, Data_path,
                                                                filename="posRatio" + str(poison_ratio) +
                                                                "lambda_" + str(lambda_x) + "_exp_" + str(j))
                        x_retrain = FO_retrain_poison(train_poison_data, train_clean_data, AG_sol_track[-1,D_x:2*D_x],
                                                    AG_sol_track[-1, 0:D_x], n_retrain, lambda_x, 0.05)
                        print("clean model learning...")
                        x_clean = FO_retrain_poison(train_poison_data, train_clean_data, AG_sol_track[-1,D_x:2*D_x],
                                                    np.zeros(D_x), n_retrain, lambda_x, 0.05)
                        x_retrain_FO = FO_retrain_poison(train_poison_data, train_clean_data, FO_sol_track[-1,D_x:2*D_x],
                                                  FO_sol_track[-1, 0:D_x], n_retrain, lambda_x, 0.05)
                        teAcc_ZOAG[iii,ii,j] = acc_for_D(x_retrain, test_data)
                        trAcc_ZOAG[iii,ii,j] = acc_for_PoisonD(x_retrain, AG_sol_track[-1,0:D_x], train_poison_data,
                                                        train_clean_data)
                        teAcc_clean[iii,ii,j] = acc_for_D(x_clean, test_data)
                        trAcc_clean[iii,ii,j] = acc_for_D(x_clean, train_data)
                        teAcc_FOAG[iii,ii,j] = acc_for_D(x_retrain_FO, test_data)
                        trAcc_FOAG[iii,ii,j] = acc_for_PoisonD(x_retrain_FO, FO_sol_track[-1,0:D_x], train_poison_data,
                                                       train_clean_data)
                    np.savez(Data_path + "/" +"MultiRatio_AG_4_2_SL_result.npz",trAcc_ZOAG=trAcc_ZOAG,teAcc_ZOAG=teAcc_ZOAG,trAcc_clean=trAcc_clean,teAcc_clean=teAcc_clean)
           
            print("trAcc_clean:")
            print(np.mean(trAcc_clean,axis=-1))
            print("teAcc_clean:")
            print(np.mean(teAcc_clean,axis=-1))
            print("trAcc_ZOAG:")
            print(np.mean(trAcc_ZOAG,axis=-1))
            print("teAcc_ZOAG:")
            print(np.mean(teAcc_ZOAG,axis=-1))
            print("trAcc_FOAG:")
            print(np.mean(trAcc_FOAG,axis=-1))
            print("teAcc_FOAG:")
            print(np.mean(teAcc_FOAG,axis=-1))
            np.savez(Data_path + "/" +"MultiRatio_AG_4_2_SL_result.npz",trAcc_ZOAG=trAcc_ZOAG,teAcc_ZOAG=teAcc_ZOAG,trAcc_FOAG=trAcc_FOAG,teAcc_FOAG=teAcc_FOAG,trAcc_clean=trAcc_clean,teAcc_clean=teAcc_clean)
    result = np.load(Data_path + "/" +"MultiRatio_AG_4_2_SL_result.npz")
    trAcc_ZOAG = result["trAcc_ZOAG"]
    teAcc_ZOAG = result["teAcc_ZOAG"]
    trAcc_FOAG = result["trAcc_FOAG"]
    teAcc_FOAG = result["teAcc_FOAG"]
    trAcc_clean = result["trAcc_clean"]
    teAcc_clean = result["teAcc_clean"]
    if lambda_x_list[0] == 0:
        lambda_x_list.remove(0)
        trAcc_ZOAG = trAcc_ZOAG[1:]
        teAcc_ZOAG = teAcc_ZOAG[1:]
        trAcc_FOAG = trAcc_FOAG[1:]
        teAcc_FOAG = teAcc_FOAG[1:]
        trAcc_clean = trAcc_clean[1:]
        teAcc_clean = teAcc_clean[1:]
    plot_lambda_one90line(lambda_x_list,np.mean(teAcc_clean,axis=-1).flatten(),"Testing accuracy","te_change_lambda_one90.pdf")
    count = np.sum(np.mean(teAcc_clean,axis=-1).flatten() > 0.9)
    plot_lambda(lambda_x_list,np.stack((np.mean(trAcc_ZOAG,axis=-1).flatten(),np.mean(trAcc_FOAG,axis=-1).flatten(),np.mean(trAcc_clean,axis=-1).flatten())),"Training accuracy",["ZO-Min-Max","FO-Min-Max","No Poison"],"tr_change_lambda.pdf")
    plot_lambda(lambda_x_list,np.stack((np.mean(teAcc_ZOAG,axis=-1).flatten(),np.mean(teAcc_FOAG,axis=-1).flatten(),np.mean(teAcc_clean,axis=-1).flatten())),"Testing accuracy",["ZO-Min-Max","FO-Min-Max","No Poison"],"te_change_lambda.pdf")
    plot_lambda(lambda_x_list,np.stack((np.mean(teAcc_ZOAG,axis=-1).flatten(),np.mean(teAcc_FOAG,axis=-1).flatten(),np.mean(teAcc_clean,axis=-1).flatten())),"Testing accuracy",["ZO-Min-Max","FO-Min-Max","No Poison"],"te_change_lambda_select.pdf",count)
    plot_lambda_90line(lambda_x_list,np.stack((np.mean(teAcc_ZOAG,axis=-1).flatten(),np.mean(teAcc_FOAG,axis=-1).flatten(),np.mean(teAcc_clean,axis=-1).flatten())),"Testing accuracy",["ZO-Min-Max","FO-Min-Max","No Poison"],"te_change_lambda_90.pdf")