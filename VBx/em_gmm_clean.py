#!/usr/bin/env python
from __future__ import print_function

# 
# Python function for leave-one-out PLDA GMM clustering
#
# $Id: em_gmm_clean.py,v 1.3 2021/01/25 22:26:26 amccree Exp amccree $
#

import numpy as np
import scipy
try:
    from scipy.misc import logsumexp
except:
    from scipy.special import logsumexp
from numpy import linalg
from numpy.linalg import eigh
from sklearn.cluster import KMeans

import logging
logger = logging.getLogger(__name__)

# Function for leave-one-out PLDA GMM clustering
def em_gmm_clean(x_in, cov_wc, cov_ac, M=7, r=0.9, num_iter=30, init_labels=None, verbose=1):

    x = x_in.copy() # Don't change input
    N = x.shape[1]
    logger.debug("EM GMM, M %d N %d" % (M,N))

    # Joint diagonalization if not already diagonalized
    if min(cov_wc.shape) != 1 and len(cov_wc.shape) != 1:
        logger.debug("Applying joint diagonalization...")
        # LDA joint diagonalization to output dimension
        d = np.linalg.matrix_rank(cov_ac)
        Ulda = form_lda(cov_wc, cov_ac, d)

        # Apply LDA to input and covariances
        x = np.dot(Ulda.T,x)
        cov_wc = np.diag(np.dot(Ulda.T,np.dot(cov_wc,Ulda)))
        cov_ac = np.diag(np.dot(Ulda.T,np.dot(cov_ac,Ulda)))
        logging.debug(str(Ulda))

    if init_labels is None:
        logging.debug("Running kmeans...")
        KM = KMeans(random_state=0,n_clusters=M,n_init=10)
        cluster_ids = KM.fit_predict(x.T)
        init_labels = cluster_ids+1

    # Initialize models with posteriors from clustering
    logging.debug("Initializing GMM...")
    #p0 = 0.0001
    p0 = 0.05/M
    #p1 = 1.0 - (M*p0)
    p1 = 1.0 - ((M-1)*p0)
    posts = p0*np.ones((M,N))
    for n in range(N):
        posts[init_labels[n]-1,n] = p1

    # OOS scoring: this should only affect numerical stability of softmax in this simplified version
    LL_oos = gauss_open_score(x, cov_ac, cov_wc)

    m = M # no loop over speakers in this version
    min_prior = 0.1/N
    logging.debug("empty cluster threshold %f" % (min_prior))

    # EM iterations
    for iter in range(num_iter):

        # Update speaker priors (GMM weights)
        prior = (np.sum(posts,1)/np.sum(posts[:]))[:,np.newaxis]
        while np.min(prior) < min_prior:
            # Empty cluster: delete it
            i2 = np.argmin(prior,0)
            posts = np.delete(posts,i2,0)
            m = posts.shape[0]
            logging.debug(" deleting empty cluster %d, now m = %d" % (i2, m))
            prior = (np.sum(posts,1)/np.sum(posts[:]))[:,np.newaxis]

        if verbose:
            logging.debug("iteration %d" % iter)
            #print prior

        p1 = posts.copy()
        log_sum = 0.0

        # EM could not use functions to allow mean recursion
        for n in range(N):

            # M-step: update models (leaving out current sample)
            p2 = p1.copy()
            p2[:,n] = 0.0
            #mu_model, cov_model = GMM_update(x, cov_wc, cov_ac, p2, r)
            cnt = np.sum(p2,axis=1)
            xsum = np.dot(p2,x.T)
            mu_model, cov_model = gmm_adapt(cnt, xsum, cov_wc, cov_ac, r)

            # E-step: partition data by ML
            # GMM alignment of this sample to model (without VB)
            #posts[:,n:n+1], ltmp = GMM_post(x[:,n:n+1], mu_model, cov_model, cov_wc, prior, LL_oos[n:n+1])
            #LLRs = gmm_score(x[:,n:n+1].T, mu_model, cov_model+cov_wc) - LL_oos[n:n+1].T
            LLRs = gmm_score(x[:,n:n+1].T, mu_model, cov_model+cov_wc)
            posts[:,n:n+1], ltmp = LL_to_post(LLRs.T,prior)
            log_sum += ltmp

        if verbose:
            logging.debug("m %d, LL %f" % (m, log_sum))

        # Compute and print new prior
        prior = (np.sum(posts,1)/np.sum(posts[:]))[:,np.newaxis]
        logging.info("Speaker priors:")
        logging.info(str(prior))

    # Return hard cluster labels (could be soft)
    lab = np.argmax(posts,0)+1
    return lab

# Function for Bayesian adaptation of Gaussian model
# Enroll type can be ML, MAP, or Bayes
def gmm_adapt(cnt, xsum, cov_wc, cov_ac, r=0, enroll_type='Bayes'):

    # Compute ML model
    cnt = np.maximum(0*cnt+(1e-10),cnt)
    mu_model = xsum / cnt[:,None]
    cov_model = 0*mu_model

    if not enroll_type == 'ML':

        # MAP adaptation
        # Determine covariance of model mean posterior distribution
        # Determine mean of model mean posterior distribution

        if r == 0:
            Nsc = 1.0/cnt
        elif r == 1:
            Nsc = 0.0*cnt+1.0
        else:
            M = cnt.shape[0]
            Nsc = np.zeros(M,)
            for m in range(M):
                Nsc[m] = 1.0/compute_Neff(cnt[m], r)

        cov_mean = cov_wc*Nsc[:,None]

        # MAP mean plus model uncertainty
        temp = cov_ac / (cov_ac + cov_mean)
        mu_model *= temp
        if enroll_type == 'Bayes':
            # Bayesian covariance of mean uncertainty
            cov_model = temp*cov_mean

    # Return
    return mu_model, cov_model

def gmm_score(X, means, covars):
    """Compute Gaussian log-density at X for a diagonal model"""

    inv_covars = 1.0/covars
    n_samples, n_dim = X.shape
    LLs = -0.5 * (- np.sum(np.log(inv_covars), 1)
                  + np.sum((means ** 2) * inv_covars, 1)
                  - 2 * np.dot(X, (means * inv_covars).T))
    LLs -= 0.5 * (np.dot(X ** 2, (inv_covars).T))
    #LLs -= 0.5 * (n_dim * np.log(2 * np.pi))

    return LLs

# Function for posteriors of GMM models
def GMM_post(x, mu_model, cov_model, cov_wc, prior, LL_oos=0):

    # Compute LLRs to open set
    num_models = len(mu_model)
    npts = x.shape[1]
    LLRs = np.zeros((num_models,npts))
    #LLRs = gmm_score(x, mu_model, cov_model+cov_wc) - LL_oos
    LLRs = gmm_score(x, mu_model, cov_model+cov_wc)
#    for m in range(num_models):
#        LLRs[m,:] = gauss_score(x, mu_model[m], cov_model[m]+cov_wc) - LL_oos

    # Compute posteriors and sum
    posts, log_sum = LL_to_post(LLRs,prior)

    # Return posteriors/LLRs and sum
    return posts, log_sum

# Function to update GMM models
def GMM_update(x, cov_wc, cov_ac, posts, r=0):

    num_models = posts.shape[0]
    N = posts.shape[1]
    mu_model = []
    cov_model = []

    # Compute stats of all models from posteriors
    cnts, xsums = gauss_stats(x, posts)

    for m in range(num_models):
        mu, cov = gauss_adapt(cnts[m], xsums[:,m], cov_wc, cov_ac, r)
        mu_model.append(mu)
        cov_model.append(cov)

    return mu_model, cov_model

# Function for Bayesian adaptation of Gaussian model
# Note: no longer computes predictive distribution, just posterior
def gauss_adapt(cnt, xsum, cov_wc, cov_ac, r=0):
    
    # Get parameters
    xdim = xsum.shape[0]

    # MAP adaptation
    # Determine covariance of model mean posterior distribution
    # Determine mean of model mean posterior distribution

    # Compute ML model
    if cnt > 1e-10:
        mu_model = xsum/cnt
    else:
        mu_model = 0*xsum
    cov_model = 0

    Neff = compute_Neff(cnt, r)
    cov_mean = cov_wc/Neff

    if 0:
        print("cnt %f, r %.2f, Neff %.2f" % (cnt,r,Neff))

    # MAP mean plus model uncertainty
    temp = cov_ac + cov_mean

    # Diagonal covariance
    mu_model *= (cov_ac/temp)

    # Bayesian covariance of mean uncertainty
    cov_model = (cov_ac/temp)*cov_mean

    # Return
    return mu_model, cov_model

# Function for effective number of counts
def compute_Neff(cnt, r):

    # Correlation model for enrollment cuts (0=none,1=single-cut scoring)
    if cnt <= 1.0:
        Neff = cnt
    else:
        Neff = (cnt*(1-r)+2*r) / (1+r)

    return Neff

# Function to compute stats of Gaussians from data and posteriors
def gauss_stats(x, posts=0):
    
    # Get parameters
    if not hasattr(posts,'shape'):
        # No posteriors: one model, use all data
        posts = np.ones((1,x.shape[1]))
    
    cnts = posts.sum(1)
    xsums = np.dot(x,posts.T)

    # Return stats
    return cnts, xsums

# Function for Gaussian scoring
def gauss_score(x, mu_model, cov_model):

    # Get parameters
    xdim = x.shape[0]
    npts = x.shape[1]

    # Form inverse test covariance for model
    inv_cov_model, logdet_cov_model = form_inv_covar_reg(cov_model)

    # Score all vectors against model
    # Vectorized version
    if hasattr(mu_model,'shape'):
        mu_model = mu_model.reshape(xdim,1)
    y = x - mu_model

    # Diagonal
    y2 = ((y.T)*inv_cov_model).T
    LL = -0.5*((y*y2).sum(axis=0) + logdet_cov_model)

    # Return log likelihood
    return LL

# Function for single cut scoring, no model (random speaker)
def gauss_open_score(x, cov_ac, cov_wc):

    # Use Bayes function with 0,cov_ac for model
    return gauss_score(x, 0, cov_ac+cov_wc)


# Form inverse covariance matrix snd logdet
def form_inv_covar_reg(A):

    # Assume diagonal matrix
    inv_A = 1.0/A
    #logdet_A = np.log(A).sum(axis=0)
    logdet_A = np.linalg.slogdet(np.diag(A[:]))[1]

    # Return 
    return inv_A, logdet_A

# Convert LLs to posteriors
def LL_to_post_old(LLRs, prior=None):

    # Get parameters
    M = LLRs.shape[0]
    npts = LLRs.shape[1]

    # Convert to posteriors 
    LRs = np.exp(LLRs)
    posts = LRs * prior

    log_sum = 0
    for n in range(npts):
        denom = posts[:,n].sum(0)
        denom = np.maximum(denom,1e-20)
        log_sum += np.log(denom)
        posts[:,n] = posts[:,n] / denom

    return posts, log_sum

# Convert LLs to posteriors
def LL_to_post(LLs, prior=None):

    # Get parameters
    M = LLs.shape[0]
    npts = LLs.shape[1]

    if prior is not None:
        log_posts = LLs + np.log(prior)
    else:
        log_posts = LLs

    # Convert to posteriors
    log_denom = logsumexp(log_posts, axis=0)
    posts = np.exp(log_posts - log_denom)
    log_sum = np.sum(log_denom)

    return posts, log_sum

# Form LDA for dimension reduction, using diagonalizing transform
# After this, cov_wc = I and cov_ac = diagonal, sorted
def form_lda(cov_wc_in, cov_ac_in, LDA_dim):

    # Diagonal regularization
    cov_lambda = 1e-6
    cov_wc = (1-cov_lambda)*cov_wc_in + cov_lambda*np.diag(np.diag(cov_wc_in))
    cov_ac = (1-cov_lambda)*cov_ac_in + cov_lambda*np.diag(np.diag(cov_ac_in))

    # Simultaneous diagonalization of wc and ac covariances
    print("Diagonalization dimension reduction from %d to %d..." %(cov_wc.shape[0], LDA_dim))

    # First compute WCCN (eigendecomposition of WC)
    (evec,eval) = eig_sort(cov_wc.astype('float64'))
    W = np.dot(evec,np.diag(1.0/np.sqrt(eval)))

    # Now compute eigendecomposition of AC in this space
    cov_ac1 = np.dot(W.T,np.dot(cov_ac.astype('float64'),W))
    (evec1,eval1) = eig_sort(cov_ac1,LDA_dim)

    # Ulda is combination
    Ulda = np.dot(W,evec1)

    return Ulda.astype(cov_wc.dtype)

# Compute largest sorted eigenvalues and eigenvectors of square matrix
def eig_sort(mat_in, rank=0):

    # Rank not specified: keep all
    d = mat_in.shape[0]
    if not rank:
        rank = d

    # Compute eigenvalues of square matrix and sort high to low
    # symmetric matrix
    eval,evec = eigh(mat_in)
    eind = list(np.argsort(eval))
    eind.reverse()

    # Save only largest
    eind = eind[0:rank]
    evec_out = evec[:,eind]
    eval = np.maximum(eval[eind],0)

    # Return 
    return evec_out, eval
