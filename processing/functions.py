import scipy
import numpy as np


def mu_trans(data, mu, norm = 'max',quantization = True):

    """
    implement the standard mu law transform
    Parameters:
    ---------------
    data: np.array, data that needs to be inverse-transform
    mu: int, scale
    norm: float, normlisation constant
    quantization: bool, quantatise to intergral, default Ture
    
    """ 
    
    if norm == 'max':
        
        data = data/float(data.max())
    else:
        data = data/norm
            
    mu_law = np.sign(data)*(np.log(1+mu*np.abs(data))/np.log(1+mu))*mu
    
    if quantization:
        return np.round(mu_law)
    else:
        return mu_law
    
    
def inverse_mu_trans(data, mu, norm, sample = False):

    """
    implement the standard inverse mu law transform
    Parameters:
    --------------- 
    data: np.array, data that needs to be inverse-transform
    mu: int, scale
    norm: float, normlisation constant
   
    """     
    if sample:        
        means = data.flatten()
        cov = np.eye(data.size)*sample        
        data = np.random.multivariate_normal(means,cov)
        
    mu = float(mu)
    data = data/mu
            
    recover = np.sign(data)*(1/mu)*((1+mu)**np.abs(data)-1)
    
    return recover*norm

def mu_itrans(data, mu, norm = 'max',quantization = True):

    """
    implement the i-mu law transform that forces the values to -1 to 1
    Parameters:
    --------------- 
    data: np.array, data that needs to be inverse-transform
    mu: int, scale
    norm: float, normlisation constant
    quantization: bool, quantatise to intergral, default Ture
    
    """ 
    
    if norm == 'max':
        
        data = data/float(data.max())*2-1
    else:
        data = data/norm*2-1
            
    mu_law = np.sign(data)*(np.log(1+mu*np.abs(data))/np.log(1+mu))*mu
    
    if quantization:
        return np.round((mu_law+mu)/2)
    else:
        return mu_law
    
    
    
def inverse_mu_itrans(data, mu, norm, sample = False):


    """
    implement the inverse i-mu law transform
    Parameters:
    --------------- 
    
    data: np.array, data that needs to be inverse-transform
    mu: int, scale
    norm: float, normlisation constant
    """ 
    
    if sample:        
        means = data.flatten()
        cov = np.eye(data.size)*sample        
        data = np.random.multivariate_normal(means,cov)
        
    mu = float(mu)
    data = data*2-mu
    data = data/mu
            
    recover = np.sign(data)*(1/mu)*((1+mu)**np.abs(data)-1)
    
    return (recover+1)*norm/2
