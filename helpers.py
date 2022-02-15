import numpy as np
import GPy

def bhattacharya(mu1,mu2,var1,var2):
    dist = 0.25*np.log(0.25*(var1/var2 + var2/var1 + 2)) + 0.25*(((mu1-mu2)**2)/(var1+var2))
    return dist

def KL(mu1,mu2,var1,var2):
    jitter = 1e-15
    dist = np.log(np.sqrt(var2/var1)) + (var1 + (mu1 - mu2)**2)/(2*var2) - 0.5 + jitter
    return dist

def get_FC_ranking(model,x):
    
    rel_bhattacharya = np.zeros(x.shape[1])
    all_rel_bhattacharya = np.zeros(x.shape[::-1])
    
    mean_real,var_real = model.predict(x)
    
    for i in range(x.shape[1]):

        temp = np.ones(x.shape)
        temp[:,i] *= 0

        perturbed_x = x*temp

        mean_new, var_new = model.predict(perturbed_x)

        rel_bhattacharya[i] = np.sqrt(np.sum(bhattacharya(mean_real.flatten(),mean_new.flatten(),var_real.flatten(),var_new.flatten())))
        all_rel_bhattacharya[i] = bhattacharya(mean_real.flatten(),mean_new.flatten(),var_real.flatten(),var_new.flatten())

    return(rel_bhattacharya, all_rel_bhattacharya)


def get_KL_ranking(model,x,delta):
    
    rel_KL = np.zeros(x.shape[1])
    
    all_rel_KL = np.zeros(x.shape[::-1])
    
    mean_real,var_real = model.predict(x)
    
    for i in range(x.shape[1]):

        temp = np.zeros(x.shape)
        temp[:,i] += delta
        #print(temp[0])
        perturbed_x = x+temp 
        
        mean_new, var_new = model.predict(perturbed_x)

        rel_KL[i] = np.sum(np.sqrt(KL(mean_real.flatten(),mean_new.flatten(),var_real.flatten(),var_new.flatten())))
        all_rel_KL[i] = KL(mean_real.flatten(),mean_new.flatten(),var_real.flatten(),var_new.flatten())
        
        temp = np.zeros(x.shape)
        temp[:,i] += -delta

        perturbed_x = x+temp 
        mean_new, var_new = model.predict(perturbed_x)
        rel_KL[i] += np.sum(np.sqrt(KL(mean_real.flatten(),mean_new.flatten(),var_real.flatten(),var_new.flatten())))
        all_rel_KL[i] += KL(mean_real.flatten(),mean_new.flatten(),var_real.flatten(),var_new.flatten())
        
        rel_KL[i] *= 0.5/delta
        all_rel_KL[i] *= 0.5/delta
        
    return (rel_KL, all_rel_KL)
