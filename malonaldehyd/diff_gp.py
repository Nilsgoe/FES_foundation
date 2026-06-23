import jax.numpy as np
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from jax import grad,jit,vmap,lax,random, jacfwd

from jax import config
config.update("jax_enable_x64", True)

class diff_GP:
    def __init__(self,delta=1.0,l=1.0,alpha_RQ=0.5,sigma=0.1,
                max_steps=401, learning_rate=1e-5, momentum=0.5, verbose = False):
        # kernel hyperparameters
        self.delta    = delta
        self.l        = l
        self.alpha_RQ = alpha_RQ
        self.sigma    = sigma
        # optimizer parameters
        self.max_steps     = max_steps
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.verbose       = verbose
    
    def optimize(self,X_train,dY_train):
        self.sigma, self.delta, self.l, self.alpha_RQ = _hyper_opt(X_train,
                                                                  dY_train,
                                                                  sigma=self.sigma,
                                                                  l=self.l,
                                                                  delta=self.delta,
                                                                  alpha_RQ=self.alpha_RQ,
                                                                  max_steps=self.max_steps,
                                                                  learning_rate=self.learning_rate,
                                                                  momentum=self.momentum,
                                                                  verbose=self.verbose)
    def train(self,X_train,dY_train):
        self.X_train = X_train
        self.alpha = _train_diff(X_train,dY_train,
                                delta=self.delta,
                                l=self.l,
                                sigma=self.sigma,
                                alpha_RQ=self.alpha_RQ)
    def predict(self,X_predict):
        Y_predict, Y_std = _predict(X_predict,self.X_train,self.alpha,
                                    delta=self.delta,
                                    l=self.l,
                                    alpha_RQ=self.alpha_RQ,
                                    sigma=self.sigma)
        return Y_predict, Y_std

    def predict_diff(self,X_predict):
        dY_predict, dY_std = _predict_diff(X_predict,self.X_train,self.alpha,
                                           delta=self.delta,
                                           l=self.l,
                                           alpha_RQ=self.alpha_RQ,
                                           sigma=self.sigma)
        return dY_predict, dY_std
    def print_hypers(self):
        print(
            f'Delta:{self.delta:.4f} Sigma:{self.sigma:.4f} lengthscale:{self.l:.4f} alpha_RQ:{self.alpha_RQ:.4f}')

def Kernel_Hess_vmap(kernel, X1, X2, delta=1.0,l=1.0,alpha_RQ=0.5):
    N_x1   = X1.shape[0]
    N_x2   = X2.shape[0]
    if len(X1.shape)>1:
        N_dim = X1.shape[1]
    else:
        N_dim = 1
    X1 = X1.reshape(N_x1,N_dim)
    X2 = X2.reshape(N_x2,N_dim)
    #k_map_x1 = vmap(kernel,in_axes=(0, None),out_axes=0)
    #k_map_x2 = vmap(k_map_x1,in_axes=(None, 0),out_axes=1)
    #K = k_map_x2(X1,X2)
    K = vmap(lambda x: vmap(lambda y: kernel(x, y, delta=delta,l=l,alpha_RQ=alpha_RQ),)(X2))(X1)
    K = K.transpose(0,2,1,3)
    K = K.reshape((N_x1*N_dim,N_x2*N_dim))
    return K

def Kernel_Grad_vmap(kernel, X1, X2, delta=1.0,l=1.0,alpha_RQ=0.5):
    N_x1   = X1.shape[0]
    N_x2   = X2.shape[0]
    if len(X1.shape)>1:
        N_dim = X1.shape[1]
    else:
        N_dim = 1
    X1 = X1.reshape(N_x1,N_dim)
    X2 = X2.reshape(N_x2,N_dim)
    #k_map_x1 = vmap(kernel,in_axes=(0, None),out_axes=0)
    #k_map_x2 = vmap(k_map_x1,in_axes=(None, 0),out_axes=1)
    #K = k_map_x2(X1,X2,delta=delta,l=l,alpha_RQ=alpha_RQ)
    K = vmap(lambda x: vmap(lambda y: kernel(x, y, delta=delta,l=l,alpha_RQ=alpha_RQ),)(X2))(X1)
    K = K.transpose(0,2,1)
    K = K.reshape((N_x1*N_dim,N_x2))
    return K

def Kernel_vmap(kernel, X1, X2, delta=1.0,l=1.0,alpha_RQ=0.5):
    N_x1   = X1.shape[0]
    N_x2   = X2.shape[0]
    if len(X1.shape)>1:
        N_dim = X1.shape[1]
    else:
        N_dim = 1
    X1 = X1.reshape(N_x1,N_dim)
    X2 = X2.reshape(N_x2,N_dim)
    #k_map_x1 = vmap(kernel,in_axes=(0, None),out_axes=0)
    #k_map_x2 = vmap(k_map_x1,in_axes=(None, 0),out_axes=1)
    #K = k_map_x2(X1,X2,delta=delta,l=l,alpha_RQ=alpha_RQ)
    K = vmap(lambda x: vmap(lambda y: kernel(y, x, delta=delta,l=l,alpha_RQ=alpha_RQ),)(X1))(X2)
    return K


def RQ_base(X1,X2,delta=1.0,l=1.0,alpha_RQ=0.5):
    d2 = np.sum((X1-X2)**2)
    return delta**2*(1. + d2/(2.*alpha_RQ*l**2))**(-alpha_RQ)

def RQ(X1, X2, delta=1.0, l=1.0, alpha_RQ=0.5, periodicity=1.0,composite_kernel=True):#composite_kernel_v1
    """Combines RQ with Periodic and Matérn Kernels."""
    rq = RQ_base(X1, X2, delta, l, alpha_RQ)
    if composite_kernel== True:
        periodic = delta**2 * np.exp(-2 * np.sin(np.pi * np.abs(X1 - X2) / periodicity)**2 / l**2)
        matern = delta**2 * (1 + np.sqrt(3) * np.abs(X1 - X2) / l) * np.exp(-np.sqrt(3) * np.abs(X1 - X2) / l)
        out = rq + periodic + matern
    else:
        return rq
    return out[0]

def composite_kernel_v2(X1, X2, delta=1.0, l=1.0, alpha_RQ=0.5, periodicity=1.0): #composite_kernel_v2
    """Combines RQ with Periodic and Matérn Kernels."""
    rq = RQ_base(X1, X2, delta, l, alpha_RQ)
    periodic = delta**2 * np.exp(-2 * np.sin(np.pi * np.abs(X1 - X2) / periodicity)**2 / l**2)
    matern = delta**2 * (1 + np.sqrt(3) * np.abs(X1 - X2) / l) * np.exp(-np.sqrt(3) * np.abs(X1 - X2) / l)
    out= (rq / np.max(rq)) + (periodic / np.max(periodic)) + (matern / np.max(matern))
    return out[0]


dRQ  = grad(RQ)

d2RQ = jacfwd(dRQ,argnums=1)

@jit
def _train_diff(x,dy,delta=1.0,l=1.0,sigma=0.3,alpha_RQ=0.5,noise_matrix=None):
    #build kernel matrix
    dy = dy.flatten()
    N_x   = x.shape[0]
    if len(x.shape)>1:
        N_dim = x.shape[1]
    else:
        N_dim =1
    N_dy  = N_dim*N_x
    #K = np.zeros((N,N))
    if noise_matrix is None:
        I = np.eye(N_dy)
        s = sigma**2*I
    else:
        I = np.eye(N_dy)
        s = noise_matrix**2 + sigma**2*I
    K = Kernel_Hess_vmap(d2RQ,x,x,delta=delta,l=l,alpha_RQ=alpha_RQ)

    # Solve linear system
    alpha = np.linalg.solve(K+s,dy)
    return alpha#, -lml

@jit
def _predict(x_p,x_t,alpha,delta=1.0,l=1.0,alpha_RQ=0.5,sigma=0.3,noise_matrix=None):   
    K_diag  = np.diagonal(
            Kernel_vmap(RQ,x_p,x_p,delta=delta,l=l,alpha_RQ=alpha_RQ))
    K_pred  = Kernel_Grad_vmap(dRQ,x_t,x_p,delta=delta,l=l,alpha_RQ=alpha_RQ)
    K_train = Kernel_Hess_vmap(d2RQ,x_t,x_t,delta=delta,l=l,alpha_RQ=alpha_RQ)
    N = K_train.shape[0]
    if noise_matrix is None:
        I = np.eye(N)
        s = sigma**2*I
    else:
        I = np.eye(N)
        s = noise_matrix**2 + sigma**2*I
    K_s = K_train+s
    L,lower = cho_factor(K_s,lower=True)
    V = solve_triangular(L,K_pred,lower=True)
    y_var = K_diag - np.einsum("ij,ji->i", V.T, V)
    y_var = np.clip(y_var, 0.0) 
    y = np.matmul(alpha,K_pred)
    #print(f"Predict K:{K.shape}")

    return y, np.sqrt(y_var)

@jit
def _log_marginal_likelihood(hypers,x,dy,noise_matrix=None):
    #hypers[0] = sigma hypers[1]=delta hypers[2]=l hypers[3]=alpha_RQ
    dy = dy.flatten()
    N = dy.shape[0]

    if noise_matrix is None:
        I = np.eye(N)
        s = hypers[0]**2*I
    else:
        I = np.eye(N)
        s = noise_matrix**2 + hypers[0]**2*I
        
    K = Kernel_Hess_vmap(d2RQ,x,x,delta=hypers[1],l=hypers[2],alpha_RQ=hypers[3])
    K_s = K+s
    
    L,lower = cho_factor(K_s,lower=True)
    alpha = cho_solve((L,lower),dy)

    term1 = -0.5*np.matmul(dy,alpha)
    term2 = -np.log(np.diag(L)).sum() #-0.5*np.log(1./np.linalg.det(K_s))
    term3 = -0.5*N*np.log(2.*np.pi)

    lml = term1+term2+term3
    return -lml

@jit
def _predict_diff(x_p,x_t,alpha,delta=1.0,l=1.0,alpha_RQ=0.5,sigma=0.3,noise_matrix=None):
    K_diag  = np.diagonal(Kernel_Hess_vmap(d2RQ,x_p,x_p,delta=delta,l=l,alpha_RQ=alpha_RQ))
    K_pred  = Kernel_Hess_vmap(d2RQ,x_t,x_p,delta=delta,l=l,alpha_RQ=alpha_RQ)
    K_train = Kernel_Hess_vmap(d2RQ,x_t,x_t,delta=delta,l=l,alpha_RQ=alpha_RQ)
    N = K_train.shape[0]
    if noise_matrix is None:
        I = np.eye(N)
        s = sigma**2*I
    else:
        I = np.eye(N)
        s = noise_matrix**2 + sigma**2*I
    K_s = K_train+s
    L,lower = cho_factor(K_s,lower=True)

    V = solve_triangular(L,K_pred,lower=True)
    dy_var = K_diag - np.einsum("ij,ji->i", V.T, V)
    dy_var = np.clip(dy_var, 0.0) 
    #print(f"Predict diff K:{K.shape}")
    dy = np.matmul(alpha,K_pred)
    
    return dy, np.sqrt(dy_var)
    #return (np.matmul(K,alpha))

def _hyper_opt(x_t,dy,
              noise_matrix=None,
              sigma=0.05,
              l=0.2,
              delta=1.0,
              alpha_RQ=0.1,
              max_steps=500,
              learning_rate=1e-3,
              momentum=0.4,
              n_outer=5,
              seed=42,
              verbose=False):
    dlml = grad(_log_marginal_likelihood)
    hyperlist = []
    lmllist = []
    key = random.PRNGKey(seed)
    
    if noise_matrix is None:
        sigma_opt = True
        delta_scale = 1.0
    else:
        sigma_opt = True
        delta_scale = 1.0 #/np.min(np.diagonal(noise_matrix))
        #print(delta_scale)
    if verbose:    
        print(f'Optimizing sigma: {sigma_opt}')
    
    for i_outer in range(n_outer):
        if verbose:
            print(f"=== Outer Optimization Loop {i_outer} ===")
        key, subkey = random.split(key)
        random_numbers = random.uniform(subkey,shape=(4,1))
        if i_outer == 0:
            hypers = np.array([sigma,delta_scale*delta,l,alpha_RQ])
        else:
            sigma = 10**(random_numbers[0,0]*2.-2.)
            delta = delta_scale*(random_numbers[1,0]+0.5)
            l = random_numbers[2,0]+0.1
            alpha_RQ = 10**(random_numbers[3,0]*2.-1.)
            hypers = np.array([sigma,delta,l,alpha_RQ])
        if sigma_opt==False:
            #print(f'resetting sigma value')
            hypers = hypers.at[0].set(1.0)
        oldstep = np.zeros_like(hypers)
    
        for i in range(max_steps):
            lml = _log_marginal_likelihood(hypers,x_t,dy,noise_matrix=noise_matrix)
            grad_lml = dlml(hypers,x_t,dy)
            if sigma_opt==False:
                #print(f'resetting sigma gradient')
                grad_lml = grad_lml.at[0].set(0.0)
            if i%20 == 0 and verbose:
                print(f'{i} Delta:{hypers[1]:.4f} Sigma:{hypers[0]:.4f} lengthscale:{hypers[2]:.4f} alpha_RQ:{hypers[3]:.4f} -lml:{lml:.4f}')
                #print(grad_lml)
            newstep = -learning_rate*(grad_lml + momentum*oldstep)
            oldstep=newstep
            hypers += newstep
        #print(f'!!! {i_outer} Delta:{hypers[1]:.4f} Sigma:{hypers[0]:.4f} lengthscale:{hypers[2]:.4f} alpha_RQ:{hypers[3]:.4f} -lml:{lml:.4f}')
        hyperlist.append(hypers)
        lmllist.append(lml)
    sigma, delta, l, alpha_RQ = hyperlist[np.argmin(np.array(lmllist))]
    return sigma, delta, l, alpha_RQ

if __name__=="__main__":
    print(RQ(1.1,1.0))
    print(dRQ(1.1,1.0))
 

