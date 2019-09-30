from __future__ import print_function
import numpy as np
import random
import time
import math
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size": 14})
plt.switch_backend("agg")  ### running for CCC
import os
import pdb

#############################################################################
##################### Sub functions for  ZO algorithm ZO AG #####################
#############################################################################
def project_inf(x,epsilon):
    x = np.greater(x,epsilon)*epsilon + np.less(x,-epsilon)*(-epsilon) \
    + np.multiply(np.multiply(np.greater_equal(x,-epsilon),np.less_equal(x,epsilon)),x)
    return x

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

def project(x,bound):
    D=len(x)
    flag=0
    for i in range(0,D):
        if x[i]<bound[i][0]:
            x[i]=bound[i][0]
            flag=1
            continue
        if x[i]>bound[i][1]:
            x[i]=bound[i][1]
            flag=1
            continue
    return x,flag

def project_f_l2(x,x_cen,epsilon):
    D=len(x)
    flag=0
    distance=np.linalg.norm(x-x_cen, ord=2)
    if distance>epsilon:
        flag=1
        x=x_cen+epsilon*(x-x_cen)/distance
    return x,flag

def ZOSIGNSGD_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q

        x_temp,flag2=project(x_opt - lr * np.sign(dx), bound)

        y_temp=func(x_temp)

        if np.isnan(y_temp) == 1:
            test = 1

        if i%10 == 0:
            print("ZOsignSGD for -likelihood: Iter = %d, obj = %3.4f" % (i, y_temp) )
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            # print('ZOsignSGD for -likelihood: Not descent direction')
            flag1=flag1+1
            if flag1%3==0:
                # step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGD(func,x0,step,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp=x_opt - lr *  dx
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag=flag+1
    return x_opt

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

def ZOPSGA_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp,flag2=project(x_opt+lr*dx,bound)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(-y_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
    return x_opt

def ZOPSGD_bounded_f(func,x0,dis_f,epsilon,step,x_cen,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size
    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp,flag2=project_f_l2(x_opt-lr*(dx),x_cen,epsilon)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(y_temp)
        if y_temp<best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            # print('ZO-PSGD: Not descent direction')
            flag1=flag1+1
    return x_opt

def ZOPSGA_bounded_f(func,x0,dis_f,epsilon,step,x_cen,lr=0.1,iter=100,Q=10):
    D=len(x0)
    x_opt=x0
    best_f=func(x0)
    sigma=1
    flag1=0
    step = np.min([1.0/iter, step]) ### perturbed input step size

    for i in range(0,iter):
        dx=np.zeros(D)
        for q in range(0,Q):
            u = np.random.normal(0, sigma, D)
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            grad=D*(func(x_opt+u*step)-func(x_opt))*u/step
            dx = dx + grad/Q
        x_temp,flag2=project_f_l2(x_opt+lr*dx,x_cen,epsilon)
        y_temp=func(x_temp)
        #print("x_opt=",end="")
        #print(x_temp)
        #print("lr=",end="")
        #print(lr)
        #print("step=",end="")
        #print(step)
        #print("loss=",end="")
        #print(-y_temp)
        if y_temp>best_f:
            best_f=y_temp
            x_opt=x_temp
        else:
            flag1=flag1+1
    return x_opt

def FO_retrain_poison(train_poison_data, train_clean_data, x_ini, delta_opt,
                     n_iters, lambda_x,  lr_x):
    n_x = len(x_ini)
    lr_x = np.min([lr_x, 1 / np.sqrt(n_x)])
    x_opt = x_ini.copy()
    num_train = np.shape(train_clean_data)[0]

    def loss_retrain(x):
        loss = loss_function_batch(delta_opt, x, train_poison_data, train_clean_data, np.arange(0, num_train).astype(int))
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

def AG_maxmin_l2_l2(func,x0,y0,step,lr,dis_fun,epsilon_x,epsilon_y,datapath,iter=20,inner_iter=1):
    x_opt=x0
    y_opt=y0
    flag=0
    best_f=-1000000
    AG_iter_res=np.zeros((iter,len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        def func_yfixed(x):
            return func(np.hstack((x,y_opt)))
        x_opt=(ZOPSGA_bounded_f(func_yfixed,x_opt,dis_fun,epsilon_x,step[0],x0,lr[0],inner_iter))
        #print("x_opt=",end="")
        #print(x_opt)
        #print("step_x=",end="")
        #print(step[0])
        #print("lr_x=",end="")
        #print(lr[0])
        def func_xfixed(y):
            return func(np.hstack((x_opt,y)))
        y_opt=ZOPSGD_bounded_f(func_xfixed,y_opt,dis_fun,epsilon_y,step[1],np.zeros(len(y0)),lr[1],inner_iter)

        temp_f=func_xfixed(y_opt)
        AG_iter_res[i]=x_opt
        if temp_f>best_f:
            best_f=temp_f
        else:
            flag=flag+1
            if flag%3==0:
                # step[0]=step[0]*0.9
                lr[0]=lr[0]*0.95
    #np.savez(datapath+"AG_time.npz",AG_time=AG_time)
    return x_opt,AG_iter_res,AG_time

def AG_maxmin_bounded_l2(train_poison_data, train_clean_data, index_batch, func,x0,y0,step,lr,dis_fun,bound_x,datapath, iter=20,inner_iter=1,max_time=100):
    x_opt=x0
    y_opt=y0
    flag=0
    best_f=-1000000
    AG_iter_res=np.zeros((iter,2*len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        AG_iter_res[i] = np.concatenate((x_opt,y_opt))
        #print("x_opt=",end="")
        #print(x_opt)
        #print("step_x=",end="")
        #print(step[0])
        #print("lr_x=",end="")
        #print(lr[0])

        def func_xfixed(y):
            return func(np.hstack((x_opt,y)), train_poison_data, train_clean_data, index_batch[i])
        y_opt=ZOPSGD(func_xfixed,y_opt,step[1],lr[1],inner_iter)

        def func_yfixed(x):
            return func(np.hstack((x,y_opt)), train_poison_data, train_clean_data, index_batch[i])
        x_opt=ZOPSGA_bounded(func_yfixed,x_opt,bound_x,step[0],lr[0],inner_iter)

        temp_f=func_yfixed(x_opt)
        if i%10 == 0:
            print("ZO-AG for Max-Min: Iter = %d, obj = %3.4f" % (i, temp_f) )

        #print(temp_f)
        if temp_f>best_f:
            best_f=temp_f
        else:
            flag=flag+1
            # step[0]=step[0]*0.9
            #lr[0]=lr[0]*0.95
        if time.time()-AG_time[0]>max_time:
            AG_time[i]=time.time()
            break
    #np.savez(datapath+"/"+"AG_time.npz",AG_time=AG_time)
    return x_opt,AG_iter_res,AG_time

def AG_minmax_batch(func,delta0,x0,index,step,lr,q_rand_num,epsilon,iter,max_time,inner_iter=1):
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

        if i%100 == 0:
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
        if time.time()-AG_time[0]>max_time:
            AG_time[i]=time.time()
            break
    return x_opt,AG_iter_res,AG_time

def AG_run_batch(func,delta0,x0,index,epsilon,step,lr,q_rand, iter,max_time,inner_iter=1):
    x_opt,AG_iter_res,AG_time=AG_minmax_batch(func,delta0,x0,index,step,lr,q_rand, epsilon,iter,max_time,inner_iter)
    return x_opt,AG_iter_res,AG_time


def AG_main_batch_SL(train_poison_data, train_clean_data,  x_ini, delta_ini,  eps_perturb, n_iters, x_gt, index_batch, lambda_x, lr_delta, lr_x, q_rand_num, Data_path, max_time,filename=None ):
    n_x = len(x_ini)
    lr_x = np.min([lr_x, 1/np.sqrt(n_x)])
    lr_delta = np.min([lr_delta, 1 / np.sqrt(n_x)])
    x0 = x_ini.copy()
    delta0 = project_inf(delta_ini, eps_perturb)

    def loss_AG(delta,x,index):
        loss=loss_function_batch(x,delta, train_poison_data, train_clean_data, index)
        return loss

    print("##################################################################")
    print("ZO-AG method")
    time_start=time.time()

    ### highlight: lr=[lr_delta,lr_x]
    x_opt,AG_iter_res,AG_time=AG_run_batch(loss_AG,delta0,x0,index_batch,eps_perturb,step=[0.001,0.001],lr=[lr_delta,lr_x],q_rand = q_rand_num , iter=n_iters,max_time=max_time,inner_iter=1)

    time_end=time.time()
    print('Time cost of ZO-AG:',time_end-time_start,"s")

    if filename==None:
        np.savez(Data_path + "/" +"ZOAG_4_2_SL.npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    else:
        np.savez(Data_path + "/" + "ZOAG_4_2_SL_"+filename+".npz",x_gt=x_gt,AG_iter_res=AG_iter_res,AG_time=AG_time)
    return x_opt, AG_iter_res, AG_time  ### x_opt, AG_iter_res = [delta_iter, x_iter]

def f(x):
    x_=x[0]
    y_=x[1]
    return -2*x_**6+12.2*x_**5-21.2*x_**4-6.2*x_+6.4*x_**3+4.7*x_**2-y_**6+11*y_**5-43.3*y_**4+10*y_+74.8*y_**3-56.9*y_**2+4.1*x_*y_+0.1*y_**2*x_**2-0.4*y_**2*x_-0.4*x_**2*y_

def distance_fun(x1,x2):
    return np.linalg.norm(x1-x2,ord=2)

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

#############################################################################
#####################            class STABLEOPT            #####################
#############################################################################

class STABLEOPT:
    def __init__(self,train_poison_data,train_clean_data,index, beta,init_num,mu0,epsilon,D,datapath,iter=100,step=[0.05,0.05],lr=[0.05,0.05],noise=0.5,max_time=1000):
        self.noise=noise
        self.theta0=1
        self.thetai=np.ones(2 * D)
        self.sigma2=1
        self.mu0=mu0
        self.init_num=init_num
        self.iter=iter
        self.t=0 
        self.beta=beta
        self.epsilon=epsilon
        self.T=len(beta)
        self.D=D
        self.x=np.zeros((self.T,2 * self.D))
        self.y=np.zeros(self.T)
        self.step=step
        self.lr=lr
        self.iter_initial_point=np.zeros(2 * self.D)
        self.datapath=datapath
        self.max_time=max_time
        self.train_poison_data = train_poison_data
        self.train_clean_data = train_clean_data
        self.index = index

    def k_(self,x1,x2):
        r=0
        for i in range(0,self.D):
            r=r+(x1-x2)[i]**2/(np.max([self.thetai[i]**2, 1e-5]))
        r=np.sqrt(r)  ## distance function
        return self.theta0**2*math.exp(-math.sqrt(5)*r)*(1+math.sqrt(5)*r+5/3*r**2)
        #return self.theta0**2*math.exp(-r)

    def k2(self,x1,x2,theta0,thetai):
        r=0
        for i in range(0,self.D):
            r=r+(x1-x2)[i]**2/(np.max([thetai[i]**2,1e-5]))
        r=np.sqrt(r)
        return theta0**2*math.exp(-math.sqrt(5)*r)*(1+math.sqrt(5)*r+5/3*r**2)
        #return self.theta0**2*math.exp(-r)

    def k_t(self,x):
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k_(x,self.x[i])
        return k

    def k_t2(self,x,theta0,thetai):
        t=self.t
        k=np.zeros(t)
        for i in range(0,t):
            k[i]=self.k2(x,self.x[i],theta0,thetai)
        return k

    def K_t(self):
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k_(self.x[i],self.x[j])
        return K

    def K_t2(self,theta0,thetai):
        t=self.t
        K=np.zeros((t,t))
        for i in range(0,t):
            for j in range(0,t):
                K[i][j]=self.k2(self.x[i],self.x[j],theta0,thetai)
        return K

    def get_value(self,x):
        return f_AG(x,self.train_poison_data, self.train_clean_data, self.index[self.t])

    def observe(self,x):
        self.x[self.t]=x
        self.y[self.t]=self.get_value(x)
        self.t=self.t+1
        return self.y[self.t-1]

    def init(self):
        if self.t == self.init_num - 1:
            self.x[self.t]=np.zeros(2*self.D)
        else:
            self.x[self.t]=np.concatenate((np.random.uniform(-2,2,self.D),np.random.normal(0,1,self.D)))
        self.y[self.t]=self.get_value(np.array(self.x[self.t]))
        self.t=self.t+1
        return 0

    def get_prior_old(self):#hyper-parameter optimization
        self.D = self.D *2 
        m=np.mean(self.y[range(0,self.t)])
        def log_likehood(x): ### maximize the loglikelihood
            theta0=x[0]
            thetai=x[range(1,self.D+1)]
            mu0=x[self.D+1]    ### prior in mu
            sigma2=x[self.D+2] ### prior in variance
            tempmatrix=self.K_t2(theta0,thetai)+sigma2*np.identity(self.t)
            try:
                inv=np.linalg.inv(tempmatrix)
            except:
                print("Singular matrix when computing prior. Small identity added.")
                inv=np.linalg.inv(tempmatrix+0.1*np.identity(self.t))
            finally:
                return -0.5*(((self.y[range(0,self.t)]-mu0).T).dot(inv)).dot(self.y[range(0,self.t)]-mu0)-0.5*math.log(abs(np.linalg.det(tempmatrix)+1e-10))#-0.5*self.t*math.log(2*math.pi)
        bound=np.zeros((self.D+3,2))
        for i in range(0,self.D+3):
            bound[i][0]=-1000000 ### unconstrained
            bound[i][1]=1000000
        bound[self.D+2][0]=1e-6
        x_opt=ZOSIGNSGD_bounded(log_likehood,np.ones(self.D+3),bound,2,1,500)
        self.theta0=x_opt[0]
        self.thetai=x_opt[range(1,self.D+1)]
        self.mu0=x_opt[self.D+1]
        self.sigma2=x_opt[self.D+2]
        self.D = self.D // 2
        return 0

    def get_prior(self):#hyper-parameter optimization, no theta0
        m=np.mean(self.y[range(0,self.t)])
        self.D = self.D * 2
        def log_likehood(x): ### maximize the loglikelihood
            thetai=x[range(0,self.D)]
            mu0=x[self.D]    ### prior in mu
            sigma2=x[self.D+1] ### prior in variance
            tempmatrix=self.K_t2(self.theta0,thetai)+sigma2*np.identity(self.t)
            try:
                inv=np.linalg.inv(tempmatrix)
            except:
                print("Singular matrix when computing prior. Small identity added.")
                inv=np.linalg.inv(tempmatrix+0.1*np.identity(self.t))
            finally:
                return 0.5*(((self.y[range(0,self.t)]-mu0).T).dot(inv)).dot(self.y[range(0,self.t)]-mu0)+0.5*math.log(abs(np.linalg.det(tempmatrix)+1e-10))#-0.5*self.t*math.log(2*math.pi)
        bound=np.zeros((self.D+2,2))
        for i in range(0,self.D+2):
            bound[i][0]=-1000000
            bound[i][1]=1000000
        bound[self.D+1][0]=1e-6
        x_opt=ZOSIGNSGD_bounded(log_likehood,np.ones(self.D+2),bound,step=1e-3,lr=1,iter=500,Q=1)
        self.thetai=x_opt[range(0,self.D)]
        self.mu0=x_opt[self.D]
        self.sigma2=x_opt[self.D+1]
        self.D = self.D // 2
        return 0

    def select_init_point(self):
        index=random.randint(0,self.t-1)
        return self.x[index][0:self.D],self.x[index][self.D:2*self.D]

    def run_onestep(self,iter):
        if self.t<self.T:
            def ucb(x):
                try:
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t))
                except:
                    print("Singular matrix when computing UCB. Small identity added.")
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t)+0.01*np.identity(self.t))
                finally:
                    mu=self.mu0+((np.array(self.k_t(x)).T).dot(inv).dot(self.y[0:self.t] -self.mu0 ))  ### prior
                    sigma_square=self.theta0**2-(np.array(self.k_t(x)).T).dot(inv).dot(np.array(self.k_t(x)))
                    if(sigma_square<0):
                        print("UCB error: sigma2=",end="")
                        print(sigma_square)
                        print("Let sigma2=0")
                        sigma_square=0
                    return mu+np.sqrt(self.beta[self.t]*sigma_square)
            def lcb(x):
                try:
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t))
                except:
                    print("Singular matrix when computing LCB. Small identity added.")
                    inv=np.linalg.inv(self.K_t()+self.sigma2*np.identity(self.t)+0.01*np.identity(self.t))
                finally:
                    mu=self.mu0+((np.array(self.k_t(x)).T).dot(inv).dot(self.y[0:self.t] - self.mu0 ))
                    sigma_square=self.theta0**2-(np.array(self.k_t(x)).T).dot(inv).dot(np.array(self.k_t(x)))
                    if(sigma_square<0):
                        print("LCB error: sigma2=",end="")
                        print(sigma_square)
                        print("Let sigma2=0")
                        sigma_square=0
                return mu-np.sqrt(self.beta[self.t]*sigma_square)
            x,delta=self.select_init_point()
            if self.t==self.init_num:
                self.iter_initial_point=np.concatenate((x,delta))
            #print(x)

            # ### exhaustive search
            # N_search = 100
            # thr_temp2 = float("-inf")
            # i_search = 0
            # delta_temp = delta + 0
            # for x_search_temp in np.random.uniform(np.array([-0.95, -0.45]),
            #                                        np.array([3.2, 4.4]),
            #                                        (iter, len(x))):
            #     x_search_i = np.resize(x_search_temp, x.shape)
            #     i_search = i_search + 1
            #     def ucb_xfixed(delta):
            #         return ucb(x_search_i+delta)
            #     #### optimize over delta for minimization
            #     # delta_temp = ZOPSGD_bounded_f(ucb_xfixed, delta_temp, distance_fun, self.epsilon, self.step[1], np.zeros(len(x)),
            #     #                          0.1, 50, 2)
            #     thr_temp = float("inf")
            #     for i_search_delta in range(0,N_search):
            #         length = np.random.uniform(0, self.epsilon)
            #         angle = np.pi * np.random.uniform(0, 2)
            #         delta_search_i = np.array([length * np.cos(angle),length * np.sin(angle)])
            #         delta_search_i = np.resize(delta_search_i, delta.shape)
            #
            #         f_val_i = ucb_xfixed(delta_search_i)
            #         if f_val_i < thr_temp:
            #             delta_temp = delta_search_i
            #             thr_temp = f_val_i
            #
            #         if i_search_delta%10 == 0:
            #             print("Search for min ucb_delta: search_time = %d, obj = %3.5f" % (i_search_delta,thr_temp))
            #
            #     f_val_i = ucb_xfixed(delta_temp)
            #
            #     if f_val_i > thr_temp2:
            #         x = x_search_i
            #         delta = delta_temp
            #         thr_temp2 = f_val_i
            #
            #     print("Max-Min-UCB: search_time = %d, obj_max_x = %3.5f" % (i_search, thr_temp2))


            #########original version##############
            for i in range(0,iter):
                def ucb_xfixed(delta):
                    return ucb(np.concatenate((x,delta)))
                #### minimization over ucb for delta
                ### ZOPSGD
                
                delta=ZOPSGD(ucb_xfixed,delta,self.step[1],0.1,1) ## 0.1 learning rate
                # ### Exhaustive search
                # thr_temp = float("inf")
                # delta_ex = delta + 0
                # for i_search in range(0,100):
                #     length = np.random.uniform(0, self.epsilon)
                #     angle = np.pi * np.random.uniform(0, 2)
                #     delta_search_i = np.array([length * np.cos(angle),length * np.sin(angle)])
                #     delta_search_i = np.resize(delta_search_i, delta.shape)
                #
                #     f_val_i = ucb_xfixed(delta_search_i)
                #     if f_val_i < thr_temp:
                #         delta_ex = delta_search_i
                #         thr_temp = f_val_i

                # print("ucb_min_delta: iter = %d, obj_min_ZO = %3.5f, obj_min_ex = %3.5f" % (i, ucb_xfixed(delta),ucb_xfixed(delta_ex)))

                #### ucb maximization
                def ucb_deltafixed(x):
                    return ucb(np.concatenate((x,delta)))
                bound_x = np.ones((self.D,2))*2
                bound_x[:,0] = -bound_x[:,0]
                ### ZOPSGA
                x=ZOPSGA_bounded(ucb_deltafixed,x,bound_x,self.step[0],0.1,1)
                # ### Exhaustive search
                # thr_temp2 = float("-inf")
                # i_search = 0
                # x_ex = x + 0
                # for x_search_temp in np.random.uniform(np.array([-0.95,-0.45]),
                #                                        np.array([3.2, 4.4]),
                #                                      (100, len(x))):
                #     x_search_i = np.resize(x_search_temp, x.shape)
                #     f_val_i = ucb_deltafixed(x_search_i)
                #     i_search = i_search + 1
                #
                #     if f_val_i > thr_temp2:
                #         x_ex = x_search_i
                #         thr_temp2 = f_val_i

                # print("ucb_max_x: iter = %d, obj_max_ZO = %3.5f, obj_max_ex = %3.5f" % (i, ucb_deltafixed(x),ucb_deltafixed(x_ex)))

                if (i%50) == 0:
                    print("ucb_max_x: iter = %d, obj_max_ZO = %3.5f" % (i, ucb_deltafixed(x)))
                    print(x)
                    print(delta)

            print("Min-LCB: ")
            def lcb_xfixed(delta):
                return lcb(np.concatenate((x,delta)))
            # delta=ZOPSGD_bounded_f(lcb,x,distance_fun,self.epsilon,self.step[1],np.zeros(len(x)),0.5,100,2) ## self.lr[1]
            delta=ZOPSGD(lcb_xfixed,delta,self.step[1],0.1,100,5) ## self.lr[1]
            ### Exhaustive search
            # thr_temp = float("inf")
            # for i_search in range(0, N_search):
            #     # delta_search_tmp = np.random.normal(0, 1, len(delta))
            #     # delta_search_i = self.epsilon * delta_search_tmp / np.linalg.norm(delta_search_tmp)
            #     length = np.random.uniform(0, self.epsilon)
            #     angle = np.pi * np.random.uniform(0, 2)
            #     delta_search_i = np.array([length * np.cos(angle), length * np.sin(angle)])
            #     delta_search_i = np.resize(delta_search_i, delta.shape)
            #
            #     f_val_i = lcb_xfixed(delta_search_i)
            #     if f_val_i < thr_temp:
            #         delta = delta_search_i
            #         thr_temp = f_val_i
            #     if i_search % 10 == 0:
            #         print("Search for min lcb_delta: search_time = %d, obj = %3.5f" % (i_search, thr_temp))
            print("lcb_x: iter = %d, obj_max_ZO = %3.5f" % (i, lcb_xfixed(delta)))
            print(x)
            print(delta)
            self.x[self.t]=np.concatenate((x,delta)) ### update samples
            #print("Selected x+delta=",end="")
            #print(x+delta)
            self.observe(self.x[self.t]) ### update observations
        else:
            print("t value error!")
            return 0

    def run(self):
        for i in range(0,self.init_num):
            self.init()
        print("Init done")
        self.iter_initial_point=self.x[self.init_num-1]
        STABLEOPT_time=np.zeros(self.T-self.init_num)
        for i in range(0,self.T-self.init_num):
            STABLEOPT_time[i]=time.time()
            print(i+1,end="")
            print("/",end="")
            print(self.T-self.init_num)
            if (i%10) == 0:
                print("Getting prior.....")
                self.get_prior()
            else:
                print("Skip updating prior")
            #self.get_prior_old()

            print("theta0=",end="")
            print(self.theta0)
            print("thetai=",end="")
            print(self.thetai)
            print("mu0=",end="")
            print(self.mu0)
            print("sigma2=",end="")
            print(self.sigma2)

            print("Get prior done")
            if self.sigma2<=0:
                print("Prior sigma invaild!")
            self.run_onestep(self.iter)
            print("!!!!!!!!!!!GP max-min iteration %d is done!!!!!!!!!" % i)
            print("####################################################################")
            if i>=1:
                if time.time()-STABLEOPT_time[0]>self.max_time:
                    STABLEOPT_time[i] = time.time()
                    break
        #np.savez(self.datapath+"/"+"GP_time.npz",GP_time=GP_time)
        self.time=STABLEOPT_time
        print("Done.")
        return 0
    def print(self):
        print("##################################################################")
        print("X:")
        print(self.x)
        print("y:")
        print(self.y)

def f_AG(x,train_poison_data, train_clean_data, index_batch):
    D=len(x)
    D1=D>>1
    x_=x[range(0,D1)]
    y_=x[range(D1,D)]
    return -loss_function_batch(y_,x_,train_poison_data, train_clean_data, index_batch)

def AG_init_point(dimension,epsilon):
    x0 = np.zeros(dimension)
    y0 = np.zeros(dimension)
    return x0,y0

def min_through_first_i_indexes(data):
    res=np.zeros(len(data))
    for i in range(1,len(data)+1):
        temp=data[0:i]
        index=np.where(temp==np.min(temp))
        res[i-1]=data[int(index[0])]
    return res

def make_iter_compare(data1,data2,name1,name2,filename,figurepath):
    p1,=plt.plot(range(0,np.shape(data1)[0]),data1)
    p2,=plt.plot(range(0,np.shape(data2)[0]),data2)
    p3,=plt.plot(range(0,np.shape(data1)[0]),np.zeros(np.shape(data1)[0]),'--',color='red')
    plt.legend([p1, p2, p3], [name1, name2,"optimal"], loc='best')
    plt.xlabel("Number of iterations")
    plt.ylabel("Regret")
    if filename:
        plt.savefig(figurepath+"/"+filename)
    plt.close()

def compare_multitimes_and_plot(dimension,epsilon,data_point_num, select_point_num, noise,datapath,figurepath,q,times,iter_STABLEOPT=500,STABLEOPT_run=True,q0=1):
    for i in range(0,times):
        filename="data_"+str(q)+"_"+str(i)
        compare(dimension,epsilon,data_point_num, select_point_num, noise,datapath,figurepath,filename,q,i,iter_STABLEOPT=iter_STABLEOPT,STABLEOPT_run=STABLEOPT_run,q0=q0)
    load_data_and_plot(epsilon,select_point_num,datapath,figurepath,q,times)

def compare(dimension,epsilon,data_point_num, select_point_num, noise,datapath,figurepath,filename,q,i,iter_STABLEOPT=500,STABLEOPT_run=True,q0=1):
    if STABLEOPT_run:
        print("##################################################################")
        print("STABLEOPT method")
        #time_start=time.time()
        optimizer=STABLEOPT(beta=4*np.ones(data_point_num+select_point_num),init_num=data_point_num,mu0=0,epsilon=epsilon,D=dimension,datapath=datapath,iter=iter_STABLEOPT,step=[0.05,0.05],lr=[0.1,0.1],noise=noise)
        optimizer.run()
        #time_end=time.time()
        #print('Time cost of STABLEOPT:',time_end-time_start,"s")
        STABLEOPT_iter_x=optimizer.x[optimizer.init_num:optimizer.T]
        STABLEOPT_time=optimizer.time

        print("##################################################################")
        x0,y0=AG_init_point(dimension,epsilon)
        x0=optimizer.x[optimizer.init_num]
    else:
        x0,y0=AG_init_point(dimension,epsilon)
        load_filename="data_"+str(q0)+"_"+str(i)+".npz"
        #print(load_filename)
        data=np.load(datapath+"/"+load_filename)
        STABLEOPT_iter_x=data["STABLEOPT_iter_x"]
        #print(STABLEOPT_iter_x)
        x0=STABLEOPT_iter_x[0]
        STABLEOPT_iter_x=[]
        STABLEOPT_time=[]
    print("AG method")
    #time_start=time.time()
    x_opt,AG_iter_x,AG_time=AG_run(f_AG,x0,y0,step=[0.05,0.05],lr=[0.1,0.1],dis_fun=distance_fun, epsilon=epsilon,datapath=datapath,iter=select_point_num,inner_iter=q)
    #print("Decision:",end="")
    #print(x_opt)
    #time_end=time.time()
    #print('Time cost of AG:',time_end-time_start,"s")
    np.savez(datapath+"/"+filename, STABLEOPT_iter_x= STABLEOPT_iter_x, AG_iter_x= AG_iter_x,STABLEOPT_time=STABLEOPT_time,AG_time=AG_time)
    

def min_ZOPSGD_multitimes(x,epsilon,times):
    min=1000000
    para=np.linspace(0.5,1.5,times)
    for j in range(0,times):
        temp=f(ZOPSGD_bounded_f(f,x,distance_fun,epsilon,0.1*para[j],x,lr=0.05*para[j],iter=200,Q=10))
        if temp<min:
            min=temp
    return min

def mean(data):
    n=len(data)
    iter=len(data[0])
    mean=np.zeros(iter)
    for i in range(0,iter):
        iter_data=np.zeros(n)
        for j in range(0,n):
            iter_data[j]=np.array(data[j][i])
        mean[i]=np.mean(iter_data)
    return mean

def load_data_and_plot_q(epsilon,select_point_num,datapath,figurepath,q,times):
    ST=[]
    AT=[]
    SY=[]
    AY=[]
    for i in range(0,times):
        filename="data_"+str(q[0])+"_"+str(i)+".npz"
        data=np.load(datapath+"/"+filename)
        STABLEOPT_iter_x=data["STABLEOPT_iter_x"]
        gt_x=data["gt_x"]
        STABLEOPT_iter_y=np.zeros(select_point_num)
        STABLEOPT_time=data["STABLEOPT_time"]
        STABLEOPT_time=STABLEOPT_time-STABLEOPT_time[0]+1

        for i in range(0,select_point_num):
            STABLEOPT_iter_y[i]=min_ZOPSGD_multitimes(STABLEOPT_iter_x[i],epsilon,5)  ### return perturbed x
        gt_y=min_ZOPSGD_multitimes(gt_x,epsilon,5) ### return perturbed x
        ST.append(STABLEOPT_time)
        SY.append(STABLEOPT_iter_y)
    ST=mean(ST)
    SY=mean(SY)

    for j in range(0,len(q)):
        AG_T=[]
        AG_Y=[]
        for i in range(0,times):
            filename="data_"+str(q[j])+"_"+str(i)+".npz"
            data=np.load(datapath+"/"+filename)
            AG_iter_x=data["AG_iter_x"]
            gt_x=data["gt_x"]
            AG_iter_y=np.zeros(select_point_num)
            AG_time=data["AG_time"]
            AG_time=AG_time-AG_time[0]+1
            for i in range(0,select_point_num):
                AG_iter_y[i]=min_ZOPSGD_multitimes(AG_iter_x[i],epsilon,5) ### return perturbed x
            gt_y=min_ZOPSGD_multitimes(gt_x,epsilon,5) ### return perturbed x
            AG_T.append(AG_time)
            AG_Y.append(AG_iter_y)
        AT.append(mean(AG_T))
        AY.append(mean(AG_Y))

    legend=[]
    for j in range(0,len(q)):
         p1,=plt.plot(range(0,select_point_num),gt_y*np.ones(select_point_num)-AY[j])
         legend.append("ZO-Min-Max "+"q="+str(q[j]))
    p2,=plt.plot(range(0,select_point_num),gt_y*np.ones(select_point_num)-SY)
    p3,=plt.plot(range(0,select_point_num),np.zeros(select_point_num),'--',color='red')
    legend.append("STABLEOPT")
    legend.append("optimal")
    plt.legend(legend, loc='best')
    plt.xlabel("Number of iterations")
    plt.ylabel("Regret")
    plt.savefig(figurepath+"/"+"each_iter.pdf")
    plt.close()

    legend=[]
    for j in range(0,len(q)):
         p1,=plt.plot(range(0,select_point_num),min_through_first_i_indexes(gt_y*np.ones(select_point_num)-AY[j]))
         legend.append("ZO-Min-Max "+"q="+str(q[j]))
    p2,=plt.plot(range(0,select_point_num),min_through_first_i_indexes(gt_y*np.ones(select_point_num)-SY))
    p3,=plt.plot(range(0,select_point_num),np.zeros(select_point_num),'--',color='red')
    legend.append("STABLEOPT")
    legend.append("optimal")
    plt.legend(legend, loc='best')
    plt.xlabel("Number of iterations")
    plt.ylabel("Regret")
    plt.savefig(figurepath+"/"+"min_former_iter.pdf")
    plt.close()

    plt.yscale('log')
    legend=[]
    for j in range(0,len(q)):
         p1,=plt.plot(range(0,select_point_num),AT[j])
         legend.append("ZO-Min-Max "+"q="+str(q[j]))
    p2,=plt.plot(range(0,select_point_num),ST)
    legend.append("STABLEOPT")
    plt.legend(legend, loc='best')
    plt.xlabel("Number of iterations")
    plt.ylabel("Total time")
    plt.savefig(figurepath+"/"+"time.pdf")
    plt.close()

def load_data_and_plot(epsilon,select_point_num,datapath,figurepath,q,times):
    STABLEOPT_T=[]
    AG_T=[]
    STABLEOPT_Y=[]
    AG_Y=[]
    for i in range(0,times):
        filename="data_"+str(q)+"_"+str(i)+".npz"
        data=np.load(datapath+"/"+filename)
        AG_iter_x=data["AG_iter_x"]
        STABLEOPT_iter_x=data["STABLEOPT_iter_x"]
        gt_x=data["gt_x"]
        STABLEOPT_iter_y=np.zeros(select_point_num)
        AG_iter_y=np.zeros(select_point_num)
        STABLEOPT_time=data["STABLEOPT_time"]
        AG_time=data["AG_time"]
        STABLEOPT_time=STABLEOPT_time-STABLEOPT_time[0]+1
        AG_time=AG_time-AG_time[0]+1

        for i in range(0,select_point_num):
            STABLEOPT_iter_y[i]=min_ZOPSGD_multitimes(STABLEOPT_iter_x[i],epsilon,5)  ### return perturbed x
            AG_iter_y[i]=min_ZOPSGD_multitimes(AG_iter_x[i],epsilon,5) ### return perturbed x
        gt_y=min_ZOPSGD_multitimes(gt_x,epsilon,5) ### return perturbed x
        STABLEOPT_T.append(STABLEOPT_time)
        AG_T.append(AG_time)
        STABLEOPT_Y.append(STABLEOPT_iter_y)
        AG_Y.append(AG_iter_y)
        
    make_iter_compare(gt_y*np.ones(select_point_num)-mean(STABLEOPT_Y),gt_y*np.ones(select_point_num)-mean(AG_Y),"STABLEOPT","ZO-Min-Max","each_iter.pdf",figurepath)
    make_iter_compare(min_through_first_i_indexes(gt_y*np.ones(select_point_num)-mean(STABLEOPT_Y)),min_through_first_i_indexes(gt_y*np.ones(select_point_num)-mean(AG_Y)),"STABLEOPT","ZO-Min-Max","min_former_iter.pdf",figurepath)
    plot_t(mean(STABLEOPT_T),mean(AG_T),figurepath)

def plot_t(STABLEOPT_time,AG_time,figurepath):
    plt.yscale('log')
    p1,=plt.plot(range(0,np.shape(STABLEOPT_time)[0]),STABLEOPT_time)
    p2,=plt.plot(range(0,np.shape(AG_time)[0]),AG_time)
    plt.legend([p1, p2], ["STABLEOPT_time","ZO-Min-Max_time"], loc='best')
    plt.xlabel("Number of iterations")
    plt.ylabel("Total time")
    plt.savefig(figurepath+"/"+"time.pdf")
    plt.close()

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

def generate_index(length,b,iter,Data_path,filename): ### general
    index=[]
    for i in range(0,iter):
        temp=np.array(random.sample(range(0,length),b)) ### mini-batch scheme
        index.append(temp)  #### generate a list
    np.savez(Data_path + "/" +"batch_index_train_4_2_SL_" + filename + ".npz",index=index)
    return index

def compare_time(train_poison_data, train_clean_data, test_data, index_batch, dimension,epsilon,data_point_num, select_point_num, noise,datapath,figurepath,filename,q,i,iter_STABLEOPT=500,STABLEOPT_run=True,q0=1,max_time =100):
    if STABLEOPT_run:
        print("##################################################################")
        print("STABLEOPT method")
        #time_start=time.time()
        optimizer=STABLEOPT(train_poison_data, train_clean_data, index_batch, beta=4*np.ones(data_point_num+select_point_num),init_num=data_point_num,mu0=0,epsilon=epsilon,D=dimension,datapath=datapath,iter=iter_STABLEOPT,step=[0.001,0.001],lr=[0.4,0.6],noise=0, max_time = max_time)
        optimizer.run()
        #time_end=time.time()
        #print('Time cost of STABLEOPT:',time_end-time_start,"s")
        STABLEOPT_iter_x=optimizer.x[optimizer.init_num - 1:optimizer.T]
        STABLEOPT_time=optimizer.time

        print("##################################################################")
        #x0,y0=AG_init_point(dimension,epsilon)
        x0=optimizer.x[optimizer.init_num-1,0:dimension]
        y0=optimizer.x[optimizer.init_num-1,dimension:2*dimension]
    else:
        x0,y0=AG_init_point(dimension,epsilon)
        try:
            load_filename="data_"+str(q0)+"_"+str(i)+".npz"
            #print(load_filename)
            data=np.load(datapath+"/"+load_filename)
            STABLEOPT_iter_x=data["STABLEOPT_iter_x"]
            #print(STABLEOPT_iter_x)
            x0=STABLEOPT_iter_x[0,0:dimension]
            y0=STABLEOPT_iter_x[0,dimension:2*dimension]
        except:
            pass
        STABLEOPT_iter_x=[]
        STABLEOPT_time=[]
    print("AG method")
    #time_start=time.time()
    lr_x = 0.05
    lr_delta = 0.02
    x_gt = 1 * np.ones(dimension)
    #temp = epsilon * np.ones((dimension,2))
    #temp[:,0] = -temp[:,0]
    #print(temp)
    x_opt,AG_iter_x,AG_time = AG_main_batch_SL(train_poison_data, train_clean_data,  y0, x0,  2, select_point_num, x_gt, index_batch, 1e-3, lr_delta, lr_x, 10, datapath,max_time)
    #print("Decision:",end="")
    #print(x_opt)
    #time_end=time.time()
    #print('Time cost of AG:',time_end-time_start,"s")
    np.savez(datapath+"/"+filename, STABLEOPT_iter_x= STABLEOPT_iter_x, AG_iter_x= AG_iter_x,STABLEOPT_time=STABLEOPT_time,AG_time=AG_time)
    #plot_t_poison_acc(STABLEOPT_time,AG_time,STABLEOPT_iter_x, AG_iter_x, train_poison_data, train_clean_data, test_data, figurepath,"train",dimension)
    #plot_t_poison_acc(STABLEOPT_time,AG_time,STABLEOPT_iter_x, AG_iter_x, train_poison_data, train_clean_data, test_data, figurepath,"test",dimension)

def load_and_plot(train_poison_data, train_clean_data, test_data, dimension,epsilon,data_point_num, select_point_num, noise,datapath,figurepath,filename,q,i,iter_STABLEOPT=500):
    result = np.load(datapath+"/"+filename+".npz")
    STABLEOPT_iter_x= result["STABLEOPT_iter_x"]
    AG_iter_x= result["AG_iter_x"]
    STABLEOPT_time= result["STABLEOPT_time"]
    AG_time= result["AG_time"]
    plot_t_poison_acc_noretrain(STABLEOPT_time,AG_time,STABLEOPT_iter_x, AG_iter_x, train_poison_data, train_clean_data, test_data, figurepath,"train",dimension)
    plot_t_poison_acc_noretrain(STABLEOPT_time,AG_time,STABLEOPT_iter_x, AG_iter_x, train_poison_data, train_clean_data, test_data, figurepath,"test",dimension)
    plot_t_poison_acc(STABLEOPT_time,AG_time,STABLEOPT_iter_x, AG_iter_x, train_poison_data, train_clean_data, test_data, figurepath,"train",dimension)
    plot_t_poison_acc(STABLEOPT_time,AG_time,STABLEOPT_iter_x, AG_iter_x, train_poison_data, train_clean_data, test_data, figurepath,"test",dimension)

def loss_function_batch(x, delta, train_poison_data, train_clean_data, index_batch):#compute loss for a batch
    lambda_x = 1e-3
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

def plot_t_poison_acc(STABLEOPT_time,AG_time,STABLEOPT_res, AG_res, train_poison_data, train_clean_data, test_data, figurepath,flag,D):
    #plt.xscale('log')
    retrain_iter = 5000
    STABLEOPT_res_temp = STABLEOPT_res[0:sum(STABLEOPT_time>0)-1]
    STABLEOPT_time_temp = STABLEOPT_time[0:sum(STABLEOPT_time>0)-1]
    STABLEOPT_time_temp = STABLEOPT_time_temp-STABLEOPT_time_temp[0]
    AG_res_temp = AG_res[0:sum(AG_time>0)]
    AG_time_temp = AG_time[0:sum(AG_time>0)]
    AG_time_temp = AG_time_temp-AG_time_temp[0]
    AG_time_temp = AG_time_temp[0:sum(AG_time_temp<=STABLEOPT_time_temp[-1])]
    AG_res_temp = AG_res_temp[0:len(AG_time_temp)]
    STABLEOPT_acc_temp = np.zeros(len(STABLEOPT_time_temp))
    AG_acc_temp = np.zeros(len(AG_time_temp)//2000+1)
    AG_time_temp2 = np.zeros(len(AG_time_temp)//2000+1)
    print(len(STABLEOPT_time_temp))
    if flag == "train":
        for i in range(0, len(STABLEOPT_time_temp)):
            x_retrain_STABLEOPT = FO_retrain_poison(train_poison_data, train_clean_data, STABLEOPT_res_temp[i,D:2*D],
                                                    STABLEOPT_res_temp[i,0:D], retrain_iter, 1e-3, 0.05)
            STABLEOPT_acc_temp[i] = acc_for_PoisonD(x_retrain_STABLEOPT, STABLEOPT_res_temp[i,0:D],train_poison_data, train_clean_data)
        for i in range(0, len(AG_time_temp),2000):
            x_retrain_AG = FO_retrain_poison(train_poison_data, train_clean_data, AG_res_temp[i,D:2*D],
                                                  AG_res_temp[i,0:D], retrain_iter, 1e-3, 0.05)
            AG_acc_temp[i//2000] = acc_for_PoisonD(x_retrain_AG, AG_res_temp[i,0:D], train_poison_data, train_clean_data)
            AG_time_temp2[i//2000] = AG_time_temp[i]
        ylabel = "Training accuracy"
    else:
        for i in range(0, len(STABLEOPT_time_temp)):
            x_retrain_STABLEOPT = FO_retrain_poison(train_poison_data, train_clean_data, STABLEOPT_res_temp[i,D:2*D],
                                                  STABLEOPT_res_temp[i,0:D], retrain_iter, 1e-3, 0.05)
            STABLEOPT_acc_temp[i] = acc_for_D(x_retrain_STABLEOPT,test_data)
        for i in range(0, len(AG_time_temp),2000):
            x_retrain_AG = FO_retrain_poison(train_poison_data, train_clean_data, AG_res_temp[i,D:2*D],
                                                  AG_res_temp[i,0:D], retrain_iter, 1e-3, 0.05)
            AG_acc_temp[i//2000] = acc_for_D(x_retrain_AG,test_data)
            AG_time_temp2[i//2000] = AG_time_temp[i]
        ylabel = "Testing accuracy"
    STABLEOPT_acc_temp[0] = AG_acc_temp[0]
    p1,=plt.semilogx(STABLEOPT_time_temp + 1,STABLEOPT_acc_temp)
    p2,=plt.semilogx(AG_time_temp2 + 1,AG_acc_temp)
    #plt.ylim(0.5,1.0)
    plt.legend([p1, p2], ["STABLEOPT","ZO-Min-Max"], loc='best')
    plt.xlabel("Optimization time (seconds)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(figurepath+"/"+ylabel+"time.pdf")
    plt.close()
    STABLEOPT_acc_temp2 = [np.min(STABLEOPT_acc_temp[0:i]) for i in range(1,len(STABLEOPT_acc_temp)+1)]
    AG_acc_temp2 = [np.min(AG_acc_temp[0:i]) for i in range(1,len(AG_acc_temp)+1)]
    p1,=plt.semilogx(STABLEOPT_time_temp + 1,STABLEOPT_acc_temp2)
    p2,=plt.semilogx(AG_time_temp2 + 1,AG_acc_temp2)
    #plt.ylim(0.5,1.0)
    plt.legend([p1, p2], ["STABLEOPT","ZO-Min-Max"], loc='best')
    plt.xlabel("Optimization time (seconds)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(figurepath+"/"+ylabel+"time_min.pdf")
    plt.close()

def plot_t_poison_acc_noretrain(STABLEOPT_time,AG_time,STABLEOPT_res, AG_res, train_poison_data, train_clean_data, test_data, figurepath,flag,D):
    #plt.xscale('log')
    STABLEOPT_res_temp = STABLEOPT_res[0:sum(STABLEOPT_time>0)-1]
    STABLEOPT_time_temp = STABLEOPT_time[0:sum(STABLEOPT_time>0)-1]
    STABLEOPT_time_temp = STABLEOPT_time_temp-STABLEOPT_time_temp[0]
    AG_res_temp = AG_res[0:sum(AG_time>0)]
    AG_time_temp = AG_time[0:sum(AG_time>0)]
    AG_time_temp = AG_time_temp-AG_time_temp[0]
    AG_time_temp = AG_time_temp[0:sum(AG_time_temp<=STABLEOPT_time_temp[-1])]
    AG_res_temp = AG_res_temp[0:len(AG_time_temp)]
    STABLEOPT_acc_temp = np.zeros(len(STABLEOPT_time_temp))
    AG_acc_temp = np.zeros(len(AG_time_temp)//2000+1)
    AG_time_temp2 = np.zeros(len(AG_time_temp)//2000+1)
    print(len(STABLEOPT_time_temp))
    if flag == "train":
        for i in range(0, len(STABLEOPT_time_temp)):
            STABLEOPT_acc_temp[i] = acc_for_PoisonD(STABLEOPT_res_temp[i, D:2 * D], STABLEOPT_res_temp[i,0:D],train_poison_data, train_clean_data)
        for i in range(0, len(AG_time_temp),2000):
            AG_acc_temp[i//2000] = acc_for_PoisonD(AG_res_temp[i, D:2 * D], AG_res_temp[i, D:2 * D], train_poison_data, train_clean_data)
            AG_time_temp2[i//2000] = AG_time_temp[i]
        ylabel = "Training accuracy"
    else:
        for i in range(0, len(STABLEOPT_time_temp)):
            STABLEOPT_acc_temp[i] = acc_for_D(STABLEOPT_res_temp[i, D:2 * D],test_data)
        for i in range(0, len(AG_time_temp),2000):
            AG_acc_temp[i//2000] = acc_for_D(AG_res_temp[i, D:2 * D],test_data)
            AG_time_temp2[i//2000] = AG_time_temp[i]
        ylabel = "Testing accuracy"
    STABLEOPT_acc_temp[0] = AG_acc_temp[0]
    p1,=plt.semilogx(STABLEOPT_time_temp + 1,STABLEOPT_acc_temp)
    p2,=plt.semilogx(AG_time_temp2 + 1,AG_acc_temp)
    #plt.ylim(0.5,1.0)
    plt.legend([p1, p2], ["STABLEOPT","ZO-Min-Max"], loc='best')
    plt.xlabel("Optimization time (seconds)")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(figurepath+"/"+ylabel+"time_noretrain.pdf")
    plt.close()

if __name__ == "__main__":
    max_time = 6000
    STABLEOPT_run = True
    flag_regenerate_data = 1
    iter_STABLEOPT = 100
    n_iters = 1000000
    flag_run=1
    
    data_point_num = 2
    select_point_num = n_iters
    train_ratio = 0.7
    n_tr = 1000
    Data_path = os.path.dirname(os.path.realpath(__file__)) + "/data_4_2"
    figurepath= os.path.dirname(os.path.realpath(__file__))+ "/Results_figures"
    D_x = 100
    x_gt = 1 * np.ones(D_x)
    sigma2 = 1
    noise_std = 1e-3
    batch_sz = 100
    eps_perturb = 2
    poison_ratio = 0.15
    Data_filename = "D_4_2_Common_" + "MultipleRatio_SL"

    
    if flag_regenerate_data:
        generate_data(n_tr, sigma2, x_gt, noise_std, Data_path, Data_filename)
    train_data, train_poison_data, train_clean_data, test_data = generate_train_and_test_data(train_ratio,
                                                                                            poison_ratio,
                                                                                            Data_path,
                                                                                            Data_filename,
                                                                                            False)
    if flag_run:
        index = generate_index(np.size(train_clean_data, 0), batch_sz, n_iters, Data_path, Data_filename)
        compare_time(train_poison_data, train_clean_data, test_data, index, D_x, 100000,data_point_num, select_point_num, 0,Data_path,figurepath,"compare_poison",q=5,i=0,iter_STABLEOPT=iter_STABLEOPT,STABLEOPT_run=STABLEOPT_run,q0=5,max_time=max_time)
    load_and_plot(train_poison_data, train_clean_data, test_data, D_x, 100000,data_point_num, select_point_num, 0,Data_path,figurepath,"compare_poison",q=5,i=0,iter_STABLEOPT=iter_STABLEOPT)