import matplotlib.pyplot as plt
import pickle as pkl
import autograd.numpy as np
from autograd import grad
from numpy.random import multivariate_normal, randint
from scipy.stats import expon, cauchy, gamma, norm, halfcauchy 

def relu(x):
    return np.maximum(0, x)

def softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

def sample_log_cauchy(loc=0., scale=5.):
    return np.exp(cauchy.rvs(loc=loc, scale=scale))

def sample_exponential(a=0.5):
    return expon.rvs(scale=a)

def sample_gamma(a=.2, b=.2):
    return gamma.rvs(a, scale=b)

# global features
X = np.linspace(-4, 4, 100)[np.newaxis].T


def fprop(tau, prev_taus, n_layers, n_hid_units, is_ResNet, batch_norm=True,):
    n_prev_taus = prev_taus.shape[0]

    prev_hidden = relu(np.dot(X, norm.rvs(size=(1, n_hid_units))))
    h = 0.

    for layer_idx in range(n_layers):
        # if not a resNet and the rest of the scales are 0, break the loop
        if layer_idx > n_prev_taus and not is_ResNet:
            break
        
        # sample weights
        sigma = 0.
        eps = norm.rvs(size=(n_hid_units, n_hid_units))

        if layer_idx < n_prev_taus: sigma = prev_taus[layer_idx]
        elif layer_idx == n_prev_taus:
            sigma = tau
            if sigma < 0: break
        w_hat = sigma * eps

        # activatiom
        a = np.dot(prev_hidden, w_hat)

        # batchnorm (no trainable params)
        if batch_norm:
            a = (a - np.mean(a, axis=0)) / np.sqrt(np.var(a, axis=0) + 10)

        if is_ResNet:
            h = h + relu(a)
        else:
            h = relu(a)
        prev_hidden = h

    w_out_hat = norm.rvs(size=(n_hid_units, 1)) * n_hid_units**(-.5)
    return np.dot(prev_hidden, w_out_hat)



def main():
    # NN architecture
    n_layers = 5
    n_hid_units = 100
    is_ResNet = False 
    
    # num samples to draw
    n_samples = 5
    
    # define KLD prior
    kld_sampler = sample_log_cauchy


    ### DRAW PredCP PRIOR SAMPLES
    print("Getting %d samples for depth %d ResNet..." %(n_samples, n_layers))

    prior_samples = np.zeros((n_layers, n_samples))
    for s_idx in range(n_samples):
        print("getting sample #%d"%(s_idx+1))
        prior_samples[:, s_idx] = sample_prior(kld_sampler, n_layers, n_hid_units, is_ResNet)

    #prior_samples = pkl.load(open("tau_samples_%dL_%dh_LC-10.pkl"%(n_layers, n_hid_units), "rb"))
    print("Prior samples:")
    print(prior_samples)
    #pkl.dump(prior_samples, open("tau_samples_%dL_%dh_LC-10.pkl"%(n_layers, n_hid_units), "wb"))


    # SAMPLE FUNCTIONS
    predCP_sample_functions = []
    base_sample_functions = []
    stdNorm_sample_functions = []
    for s_idx in range(n_samples):
        predCP_sample_functions.append(fprop(None, prior_samples[:, s_idx], n_layers, n_hid_units, is_ResNet))
        #stdNorm_sample_functions.append(fprop(None, halfcauchy.rvs(loc=0, scale=1, size=n_layers), n_layers, n_hid_units, is_ResNet)) 
        #fprop(None, np.ones((n_layers,)) * .5, n_layers, n_hid_units, is_ResNet))

    #print(predCP_sample_functions)
    pkl.dump(predCP_sample_functions, open("function_prior_visuals/predCP-LC5_samples_%dh_L-%d.pkl"%(n_hid_units, n_layers), "wb"))  
        
        
    # visualize samples
    plt.figure(figsize=(4.2,3.5))

    for idx in range(n_samples):
        #plt.plot(X, stdNorm_sample_functions[idx], 'k-', linewidth=5., alpha=.3, zorder=0)
        plt.plot(X, predCP_sample_functions[idx], 'r-', linewidth=5., alpha=.7, zorder=10) 

    # dummy lines for legend
    plt.plot(X, [0 for x in range(X.shape[0])], 'k--', linewidth=1., alpha=1.)
    plt.plot(X, [-1000 for x in range(X.shape[0])], 'k-', linewidth=5., alpha=.9, label="Std. Normal")
    plt.plot(X, [-1000 for x in range(X.shape[0])], 'r-', linewidth=5., alpha=.9, label="LC-PredCP")

    plt.xlim([-4, 4])
    plt.ylim([-1, 1])
    plt.ylabel(r"$Y$", fontsize=20)
    plt.xlabel(r"$X$", fontsize=20)

    plt.legend(loc=2, fontsize=12, frameon=True)                                                                                                                                              
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.show()



def sample_prior(kld_sampler, n_layers, n_hid_units, is_ResNet, n_inv_steps=20, alpha=.00005):

    ### DEFINE FUNCTIONS

    # define E[KLD] function
    def expected_KLD(log_tau, prev_log_taus, n_layers, mc_samples=3, sigma2_y=1.):
        tau = softplus(log_tau)
        prev_taus = softplus(prev_log_taus)
        
        kld_accum = 0.
        for s_idx in range(mc_samples):
            if is_ResNet:
                f0 = fprop(0., prev_taus, n_layers, n_hid_units, is_ResNet)
            else:
                f0 = fprop(-1, prev_taus, n_layers, n_hid_units, is_ResNet)
            f1 = fprop(tau, prev_taus, n_layers, n_hid_units, is_ResNet)
            kld_accum += np.mean((f0 - f1)**2 / (2*sigma2_y))

        return kld_accum / mc_samples

    # define grad
    dEKLD_dTau = grad(expected_KLD)

    
    ### RUN ITERATIVE SAMPLING
    log_tau_samples = np.random.uniform(low=-2, high=-1, size=(n_layers,))

    for layer_idx in range(n_layers):
        k_hat = kld_sampler()
        
        for t_idx in range(n_inv_steps):

            ekld = expected_KLD(log_tau=log_tau_samples[layer_idx], prev_log_taus=log_tau_samples[:layer_idx], n_layers=n_layers)
            if not np.isfinite(ekld):
                continue
            
            ekld_prime = dEKLD_dTau(log_tau_samples[layer_idx], log_tau_samples[:layer_idx], n_layers)
            if not np.isfinite(ekld_prime):
                continue
            
            if np.abs(ekld_prime) < .1:
                ekld_prime = np.sign(ekld_prime) * .1
            
            log_tau_samples[layer_idx] = log_tau_samples[layer_idx] - alpha / (ekld_prime) * (ekld - k_hat)

    return softplus(log_tau_samples)



if __name__ == '__main__':
    main()
