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

#############################################################################
##################### Sub functions for  ZO algorithm ZO AG #####################
#############################################################################

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
            if flag%2==0:
                # step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGA(func,x0,step,lr=0.1,iter=100,Q=10):
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
        x_temp=x_opt+lr*dx
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
            flag=flag+1
            if flag%2==0:
                # step=step*0.95
                lr=lr*0.95
    return x_opt

def ZOPSGD_bounded(func,x0,bound,step,lr=0.1,iter=100,Q=10):
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
        x_temp,flag2=project(x_opt-lr*dx,bound)
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
            flag1=flag1+1
            if flag1%3==0:
                step=step*0.95
                lr=lr*0.95
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
            if flag1%3==0:
                step=step*0.95
                lr=lr*0.95
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
            if flag1%3==0:
                step=step*0.95
                lr=lr*0.95
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
            if flag1%3==0:
                step=step*0.95
                lr=lr*0.95
    return x_opt

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

def AG_maxmin_bounded_l2(func,x0,y0,step,lr,dis_fun,bound_x,epsilon_y,datapath,iter=20,inner_iter=1):
    x_opt=x0
    y_opt=y0
    flag=0
    best_f=-1000000
    AG_iter_res=np.zeros((iter,len(x0)))
    AG_time=np.zeros(iter)
    for i in range(0,iter):
        AG_time[i]=time.time()
        AG_iter_res[i] = x_opt
        #print("x_opt=",end="")
        #print(x_opt)
        #print("step_x=",end="")
        #print(step[0])
        #print("lr_x=",end="")
        #print(lr[0])

        def func_xfixed(y):
            return func(np.hstack((x_opt,y)))
        y_opt=ZOPSGD_bounded_f(func_xfixed,y_opt,dis_fun,epsilon_y,step[1],np.zeros(len(y0)),lr[1],inner_iter)

        def func_yfixed(x):
            return func(np.hstack((x,y_opt)))
        x_opt=ZOPSGA_bounded(func_yfixed,x_opt,bound_x,step[0],lr[0],inner_iter)

        temp_f=func_yfixed(x_opt)
        if i%10 == 0:
            print("ZO-AG for Max-Min: Iter = %d, obj = %3.4f" % (i, temp_f) )

        #print(temp_f)
        if temp_f>best_f:
            best_f=temp_f
        else:
            flag=flag+1
            if flag%3==0:
                # step[0]=step[0]*0.9
                lr[0]=lr[0]*0.95
    #np.savez(datapath+"/"+"AG_time.npz",AG_time=AG_time)
    return x_opt,AG_iter_res,AG_time

def AG_run(func,x0,y0,step,lr,dis_fun,epsilon,datapath,iter=20,inner_iter=2):
    D_x=len(x0)
    #D_y=len(y0)
    bound_x=np.ones((D_x,2))
    #bound_y=epsilon*np.ones((D_y,2))
    bound_x[0,:]=[-0.95,3.2]
    bound_x[1,:]=[-0.45,4.4]
    #bound_y[:,0]=-bound_y[:,0]
    x_opt,AG_iter_res,AG_time=AG_maxmin_bounded_l2(func,x0,y0,step,lr,dis_fun,bound_x,epsilon,datapath,iter,inner_iter)
    return x_opt,AG_iter_res,AG_time

def f(x):
    x_=x[0]
    y_=x[1]
    return -2*x_**6+12.2*x_**5-21.2*x_**4-6.2*x_+6.4*x_**3+4.7*x_**2-y_**6+11*y_**5-43.3*y_**4+10*y_+74.8*y_**3-56.9*y_**2+4.1*x_*y_+0.1*y_**2*x_**2-0.4*y_**2*x_-0.4*x_**2*y_

def distance_fun(x1,x2):
    return np.linalg.norm(x1-x2,ord=2)

#############################################################################
#####################            class STABLEOPT            #####################
#############################################################################

class STABLEOPT:
    def __init__(self,beta,init_num,mu0,epsilon,D,datapath,iter=100,step=[0.05,0.05],lr=[0.05,0.05],noise=0.5):
        self.noise=noise
        self.theta0=1
        self.thetai=np.ones(D)
        self.sigma2=1
        self.mu0=mu0
        self.init_num=init_num
        self.iter=iter
        self.t=0 
        self.beta=beta
        self.epsilon=epsilon
        self.T=len(beta)
        self.D=D
        self.x=np.zeros((self.T,self.D))
        self.y=np.zeros(self.T)
        self.step=step
        self.lr=lr
        self.iter_initial_point=np.zeros(self.D)
        self.datapath=datapath

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
        return f(x)+np.random.normal(0,self.noise)

    def observe(self,x):
        self.x[self.t]=x
        self.y[self.t]=self.get_value(x)
        self.t=self.t+1
        return self.y[self.t-1]

    def init(self):
        x=[]
        x.append(random.uniform(-0.95,3.2))
        x.append(random.uniform(-0.45,4.4)) 
        self.x[self.t]=np.array(x)
        self.y[self.t]=self.get_value(np.array(x))
        self.t=self.t+1
        return 0

    def get_prior_old(self):#hyper-parameter optimization
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
        return 0

    def get_prior(self):#hyper-parameter optimization, no theta0
        m=np.mean(self.y[range(0,self.t)])
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
        return 0

    def select_init_point(self):
        index=random.randint(0,self.t-1)
        return self.x[index]

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
            x=self.select_init_point()
            if self.t==self.init_num:
                self.iter_initial_point=x
            #print(x)
            delta=np.zeros(self.D)


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
                    return ucb(x+delta)
                #### minimization over ucb for delta
                ### ZOPSGD
                delta=ZOPSGD_bounded_f(ucb_xfixed,delta,distance_fun,self.epsilon,self.step[1],np.zeros(len(x)),0.1,1,5) ## 0.1 learning rate
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
                    return ucb(x+delta)

                ### ZOPSGA
                x=ZOPSGA_bounded(ucb_deltafixed,x,[[-0.95,3.2],[-0.45,4.4]],self.step[0],0.1,1,5)
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
                    print(distance_fun(delta,[0,0]))

            print("Min-LCB: ")
            def lcb_xfixed(delta):
                return lcb(x + delta)
            # delta=ZOPSGD_bounded_f(lcb,x,distance_fun,self.epsilon,self.step[1],np.zeros(len(x)),0.5,100,2) ## self.lr[1]
            delta=ZOPSGD_bounded_f(lcb_xfixed,delta,distance_fun,self.epsilon,self.step[1],np.zeros(len(x)),0.1,100,3) ## self.lr[1]
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

            self.x[self.t]=x+delta ### update samples
            #print("Selected x+delta=",end="")
            #print(x+delta)
            self.observe(x+delta) ### update observations
        else:
            print("t value error!")
            return 0

    def run(self):
        for i in range(0,self.init_num):
            self.init()
        print("Init done")
        self.iter_initial_point=self.x[self.init_num]
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

def f_AG(x):
    noise=0.5
    D=len(x)
    D1=D>>1
    x_=x[range(0,D1)]
    y_=x[range(D1,D)]
    return f(x_+y_)+np.random.normal(0,noise)

def AG_init_point(dimension,epsilon):
    x0=np.zeros(dimension)
    y0=np.zeros(dimension)
    x0[0]=random.uniform(-0.95,3.2)
    x0[1]=random.uniform(-0.45,4.4)
    for i in range(0,dimension):
        y0[i]=random.uniform(-epsilon,epsilon)
        #y0[i]=0
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
        STABLEOPT_iter_x=data["STABLEOPT_iter_x"]
        STABLEOPT_time=data["STABLEOPT_time"]
    print("AG method")
    #time_start=time.time()
    x_opt,AG_iter_x,AG_time=AG_run(f_AG,x0,y0,step=[0.05,0.05],lr=[0.1,0.1],dis_fun=distance_fun, epsilon=epsilon,datapath=datapath,iter=select_point_num,inner_iter=q)
    #print("Decision:",end="")
    #print(x_opt)
    #time_end=time.time()
    #print('Time cost of AG:',time_end-time_start,"s")
    gt_x=[-0.195,0.284]
    np.savez(datapath+"/"+filename, STABLEOPT_iter_x= STABLEOPT_iter_x, AG_iter_x= AG_iter_x,STABLEOPT_time=STABLEOPT_time,AG_time=AG_time, gt_x= gt_x)
    

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

if __name__=="__main__":
    my_path = os.path.dirname(os.path.realpath(__file__))
    datapath=my_path+"/data_applendix"
    figurepath=my_path+"/figure_applendix"
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    if not os.path.exists(figurepath):
        os.makedirs(figurepath)
    random.seed(212)
    select_point_num=50
    iter_STABLEOPT=500
    times=5
    q=[1,5,10,20]
    compare_multitimes_and_plot(dimension=2,epsilon=0.5,data_point_num=1,select_point_num=select_point_num,noise=0.5,datapath=datapath,figurepath=figurepath,q=q[0],times=times,iter_STABLEOPT=iter_STABLEOPT,STABLEOPT_run=True,q0=q[0])
    for i in range(1,len(q)):
        compare_multitimes_and_plot(dimension=2,epsilon=0.5,data_point_num=1,select_point_num=select_point_num,noise=0.5,datapath=datapath,figurepath=figurepath,q=q[i],times=times,iter_STABLEOPT=iter_STABLEOPT,STABLEOPT_run=False,q0=q[0])
    print("Start plotting...")
    load_data_and_plot_q(epsilon=0.5,select_point_num=select_point_num,datapath=datapath,figurepath=figurepath,q=q,times=times)
