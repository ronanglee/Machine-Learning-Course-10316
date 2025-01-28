import numpy as np

def func_mat(x, xx, func, par):
    return np.array([[func(x1, x2, par) for x1 in x] for x2 in xx])

def func_vec(x, func, par):
    return np.array([func(x1, x1, par) for x1 in x])

def gauss_kernel(x1, x2, par):
    length, k0 = np.array(par)
    return k0*np.exp(-sum((x1-x2)**2)/(2*length**2))

def gauss_deriv(x1, x2, par):
    length, k0 = par
    
    if hasattr(x1, '__len__') or hasattr(x2, '__len__'):
        X_arg = ((x1-x2)**2).sum()
    else:
        X_arg = (x1-x2)**2
    
    dlength = k0 * X_arg/length**3 * np.exp(-X_arg/(2*length**2))
    dk0 = np.exp(-X_arg/(2*length**2))
    
    return np.array([dlength, dk0])

def multi_gauss_kernel(x1, x2, par):
    k0 = par[0]
    lengths = np.array(par[1:])
    
    return k0*np.exp(-sum((x1-x2)**2/(2*lengths**2)))

def multi_gauss_deriv(x1, x2, par):
    k0 = par[0]
    lengths = np.array(par[1:])
    
    dlengths = k0 *  np.array([sum((x1-x2)**2/length**3) for length in lengths]) * np.exp(-sum((x1-x2)**2/(2*lengths**2)))
    dk0 = np.array([np.exp(-sum((x1-x2)**2/(2*lengths**2)))])
    return np.hstack((dk0, dlengths))
    

class Path_Tracker:
    def __init__(self):
        self.x_steps = []
        self.nloglike = []
        self.likelihood = None
    
    def get_prediction(self, burnin = 0):
        loglike_array = np.min(self.nloglike[burnin:]) -  np.array(self.nloglike[burnin:])  #shifts values as only ratio matters
        likelihood = np.exp(loglike_array)
        
        # if np.linalg.norm(likelihood) < 1e-15:
        #     likelihood = likelihood + 1e-15
        likelihood = likelihood / sum(likelihood)
        self.likelihood = likelihood
        
        return np.array([x*p for x, p in zip(self.x_steps[burnin:], likelihood[burnin:])]).sum(axis = 0)

class Gauss_Kernel:
    def __init__(self):
        self.kernel = gauss_kernel
        self.deriv = gauss_deriv
        
class Multi_Gauss_Kernel:
    def __init__(self):
        self.kernel = multi_gauss_kernel
        self.deriv = multi_gauss_deriv
        

class Gaussian_Process: 
    kernel = None
    noise = None
    hyper_par = None
    prior_func = None
    post_cov = None
    cov_eval = None
    cov_evec = None
    inv_cov = None
    x_dat = None
    t_dat = None
    
    def __init__(self, kernel = Gauss_Kernel(), hyper_par = [0.5,1], prior_func = lambda x : 0.0, noise = 1e-1, x_dat = None, t_dat = None):
        self.kernel = kernel
        self.noise = noise
        self.hyper_par = np.array(hyper_par)
        self.prior_func = prior_func
        if hasattr(t_dat, '__len__'):
            self.x_dat = np.array(x_dat).reshape(len(t_dat), -1)
            self.t_dat = np.array(t_dat)
        else:
            self.x_dat = np.array(x_dat).reshape(1, -1)
            self.t_dat = np.array(t_dat)
    
    def copy(self):
        #Work in progress... maybe stuff about "from_dict"?
        
        params = {key: vals for key, vals in self.__dict__.items() if not (hasattr(vals, '__call__') or '__' in key) }
        return params
    
    def get_inv_cov(self):
        return self.cov_evec@np.diag(self.cov_eval**(-1))@(self.cov_evec.T)
    
    def prior_cov(self, x_eval):
        return func_mat(x_eval, x_eval, self.kernel.kernel, self.hyper_par)
    
    def calculate(self, x, t): #Initializes the costly values
        if hasattr(t, '__len__'):
            self.x_dat = np.array(x).reshape(len(t), -1)
            self.t_dat = np.array(t)
        else:
            self.x_dat = np.array([x]).reshape(1, -1)
            self.t_dat = np.array(t)
        kernel_cov = func_mat(self.x_dat, self.x_dat, self.kernel.kernel, self.hyper_par)
        if len(kernel_cov.shape) > 2:
            print(self.hyper_par)
        cov = kernel_cov + self.noise**2 * np.eye(len(x))
        self.post_cov = cov
        self.cov_eval, self.cov_evec = np.linalg.eigh(cov)
        self.inv_cov = self.get_inv_cov()
    
        # return float((inv_cov@self.post_cov).sum()) - float(len(x))
    
    def predict(self, x_eval, prior = False):
        if prior:
            return self.prior_func(x_eval)

        else:
            if (self.post_cov is None):
                try:
                    self.calculate(self.x_dat, self.t_dat)
                except:
                    raise AssertionError("The calculator contains no data or covariance matrix. Only a prior can be returned")
                
            hybrid_cov = func_mat(self.x_dat, x_eval, self.kernel.kernel, self.hyper_par)
            mean = self.prior_func(x_eval) + hybrid_cov@self.inv_cov@(self.t_dat - self.prior_func(self.x_dat))
            
            return mean
    
    def uncer(self, x_eval):
        hybrid_cov = func_mat(self.x_dat, x_eval, self.kernel.kernel, self.hyper_par)
        base_term = func_vec(x_eval, self.kernel.kernel, par = self.hyper_par) #Bad naming, but it is a basic term
        post_term = ((hybrid_cov@self.inv_cov)*hybrid_cov).sum(axis = 1)
        
        return np.sqrt(base_term - post_term)*np.sqrt(len(self.t_dat))
    
    def uncer_old(self, x_eval):
        hybrid_cov = func_mat(self.x_dat, x_eval, self.kernel.kernel, self.hyper_par)
        prior_term = func_vec(x_eval, self.kernel.kernel, par = self.hyper_par)
        post_term = np.array([hybrid_cov[i].T@self.inv_cov@(hybrid_cov[i]) #There should be a better way but hey
                              for i in range(len(hybrid_cov[:,0]))])
        
        return np.sqrt(prior_term - post_term)*np.sqrt(len(self.t_dat))
    
    def loglike_hpar(self, calc = False):
        if calc:
            self.calculate(self.x_dat, self.t_dat) #sets the covariance
        
        #Determinant
        det_term = np.log(abs(self.cov_eval) + 1e-14).sum()
        
        #The term containing the inverse
        inv_term = (self.t_dat - self.prior_func(self.x_dat)).T@self.inv_cov@\
                    (self.t_dat - self.prior_func(self.x_dat))
        
        #A term scaling with the size of the data - the more data, the sharper a distribution
        size_term = len(self.t_dat)*np.log(2*np.pi)
        
        return -0.5 *(det_term + inv_term + size_term)
    
    def deriv_loglike_hpar(self, calc = False):
        if calc:
            self.calculate(self.x_dat, self.t_dat) #sets the covariance
        
        if hasattr(self.t_dat, '__len___'):
            self.x_dat = self.x_dat.reshape(len(self.t_dat, -1))
        deriv_mat = func_mat(self.x_dat, self.x_dat, self.kernel.deriv, self.hyper_par)
        delta_t = (self.t_dat - self.prior_func(self.x_dat))
        return np.array([-0.5*sum(np.diag(self.inv_cov@(deriv_mat[:,:,i]))) 
                + 0.5 * delta_t.T@self.inv_cov@(deriv_mat[:,:,i])@self.inv_cov@delta_t 
                for i in range(len(self.hyper_par))])
    
    def optimize_par(self, method, **kwargs):
        #Method returns parameters and iterations
        
        def nloglike(par):
            GP = Gaussian_Process(self.kernel, par, self.prior_func, self.noise, self.x_dat, self.t_dat)
            negloglike = -GP.loglike_hpar(calc = True)
            return negloglike
        
        def deriv_loglike(par): #Negative gradient of negative loglike
            GP = Gaussian_Process(self.kernel, par, self.prior_func, self.noise, self.x_dat, self.t_dat)
            deriv = np.array(GP.deriv_loglike_hpar(calc = True))
            return deriv
        
        def nlog_deriv(par):
            GP = Gaussian_Process(self.kernel, par, self.prior_func, self.noise, self.x_dat, self.t_dat)
            negloglike = -GP.loglike_hpar(calc = True)
            deriv = np.array(GP.deriv_loglike_hpar())
            return negloglike, deriv
                    
        par_opt, neglog, iter = method(self.hyper_par, nlog_deriv, **kwargs)
        
        return par_opt, neglog, iter
    
    def find_minimum(self, min_coords, max_coords, prec): #Make a better one
        x, y = np.linspace(min_coords, max_coords, int(1/prec) + 1).T
        X, Y = np.meshgrid(x, y)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        eval_coords = np.vstack((X_flat, X_flat)).T
        values = self.predict(eval_coords) - self.uncer(eval_coords) #Subtracts uncertainty to find the potentially smallest
        min_index = np.argmin(values)
        min_position = np.array([X_flat[min_index], Y_flat[min_index]]).reshape(1,-1)
        
        uncer_min = self.uncer(min_position)
        
        return min_position, uncer_min
        
    
    def iterative_minimum(self, min_coords, max_coords, heavy_func, method, prec, max_steps = 100):
        current_prec = 1
        add_coords, uncer = self.find_minimum(min_coords, min_coords, 100) #initial point
        add_val = heavy_func(add_coords)
        for i in range(max_steps):
            new_min_coords = add_coords - uncer
            new_max_coords = add_coords + uncer
            width_1D = np.max(abs(new_max_coords - new_min_coords))
            self.x_dat = np.vstack((self.x_dat, add_coords)).reshape(-1, len(min_coords))
            self.t_dat = np.hstack((self.t_dat, add_val)).reshape(-1,)
            par, iter = self.optimize_par(method, prec)
            self.hyper_par = par
            self.calculate(self.x_dat, self.t_dat)
            find_prec = prec/width_1D if prec/width_1D > 1e-2 else 1e-2 #May never converge...
            add_coords, uncer = self.find_minimum(new_min_coords, new_max_coords, find_prec)
            current_prec = np.max(np.sqrt(find_prec**2 + uncer**2))
            if current_prec < prec:
                return add_coords, uncer, current_prec
        return add_coords, uncer, np.inf
            
        