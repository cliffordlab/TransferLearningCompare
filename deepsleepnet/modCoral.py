"""
Correlation alignment.
Reference:
    Sun, B., Feng, J., & Saenko, K. (2016, March).
    Return of frustratingly easy domain adaptation.
    In Thirtieth AAAI Conference on Artificial Intelligence.
:author: Vincent Vercruyssen (2019)
:license: Apache License, Version 2.0, see LICENSE for details.
"""

import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.base import BaseEstimator

from transfertools.models.base import BaseDetector
from transfertools.utils.preprocessing import TransferScaler


# ----------------------------------------------------------------------------
# CORAL class
# ----------------------------------------------------------------------------

class RegCORAL(BaseEstimator, BaseDetector):
    """ Correlation alignment algorithm.
    MODIFIED VERSION of code obtained from https://github.com/Vincent-Vercruyssen/transfertools
    Parameters
    ----------
    scaling : str (default='standard')
        Scale the source and target domain before transfer.
        Standard scaling is indicated in the paper.
    Attributes
    ----------
    type_ : str
        The type of transfer learning (e.g., domain adaptation).
    X_trans_ : np.array of shape (<= n_samples,)
        The (transformed) source instances that are transferred.
    Ixs_trans_ : np.array of shape (n_samples, n_features)
        The indices of the instances selected for transfer.
    """

    def __init__(self,
                 scaling='standard',
                 tol=1e-8,
                 verbose=False):
        super().__init__(
            scaling=scaling,
            tol=tol,
            verbose=verbose)
        
        # type
        self.type_ = 'domain_adaptation'

    def fit(self, Xs=None, Xt=None, ys=None, yt=None, reg = 100.0):
        """ Fit the model on data X.
        Parameters
        ----------
        Xs : np.array of shape (n_samples, n_features), optional (default=None)
            The source instances.
        Xt : np.array of shape (n_samples, n_features), optional (default=None)
            The target instances.
        ys : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the source instances.
        yt : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the target instances.
        reg: Regularization parameter to add to the diagonal elements of the
            covariance matrix when computing the whitening transformation. (default=1.0)
        Returns
        -------
        self : object
        """
        
        # check all inputs
        Xs, Xt, ys, yt = self._check_all_inputs(Xs, Xt, ys, yt)
        
        ns, nfs = Xs.shape
        nt, nft = Xt.shape
        
        # align means: feature normalization/standardization!
        self.target_scaler_ = TransferScaler(self.scaling)
        self.source_scaler_ = TransferScaler(self.scaling)
        Xt = self.target_scaler_.fit_transform(Xt)
        Xs = self.source_scaler_.fit_transform(Xs)
        
        # align covariances: denoising - noising transformation
        Cs = np.cov(Xs.T) + reg*np.eye(nfs)
        Ct = np.cov(Xt.T) + reg*np.eye(nft)
        csp = sp.linalg.fractional_matrix_power(Cs, -1/2)
        ctp = sp.linalg.fractional_matrix_power(Ct, 1/2)
        self.A_ = np.dot(csp, ctp)
        
        # transferred source instances
        self.Xt_trans_ = Xt
        self.Xs_trans_ = np.dot(Xs, self.A_).real
        self.Ixs_trans_ = np.arange(0, ns, 1)
        
        return self

    def transfer(self, Xs, ys=None):
        """ Apply transfer to the source instances.
        
        Parameters
        ----------
        Xs : np.array of shape (n_samples, n_features)
            The source instances.
        ys : np.array of shape (n_samples,), optional (default=None)
            The ground truth of the source instances.
        return_indices : bool, optional (default=False)
            Also return the indices of the source instances
            selected for transfer.
        Returns
        -------
        Xs_trans : np.array of shape (n_samples, n_features)
            The (transformed) source instances after transfer.
        Ixs_trans : np.array of shape (<= n_samples,)
            The indices of the source instances selected for transfer.
        """
        
        #ns, _ = Xs.shape
        if len(Xs.shape) <= 1: Xs = Xs[None,...]

        # scaling
        Xs = self.source_scaler_.transform(Xs)

        # transform
        Xs_trans = np.dot(Xs, self.A_).real

        return Xs_trans

class SCORAL():
    """
    Supervised CORAL. Performs CORAL transform independently on each class.
    """
    perClassTransforms = []
    def fit(self, Xs=None, Xt=None, ys=None, yt=None, reg = 10.0):
        
        classLabels = np.unique(ys)
        self.perClassTransforms = {label:RegCORAL() for label in classLabels} # Initialize one transform per class
        for label in classLabels:
            
            # Obtain samples from source and target from same class
            Xs_i = Xs[ys == label]
            Xt_i = Xt[yt == label]
            
            # Randomly sample from the larger sample set so that both are the same size
            if Xs_i.shape[0] > Xt_i.shape[0]:
                Xs_i = Xs_i[np.random.choice(Xs_i.shape[0], Xt_i.shape[0], replace=False), :]
            elif Xs_i.shape[0] < Xt_i.shape[0]:
                Xt_i = Xt_i[np.random.choice(Xt_i.shape[0], Xs_i.shape[0], replace=False), :]
                
            # Train each transform
            self.perClassTransforms[label].fit(Xs=Xs_i, Xt=Xt_i, reg = reg)

    def transfer(self, Xs, ys):
        """
        Transform elements from each class using the CORAL transform trained on elements from that class
        """
        if Xs.shape[0] > 1:
            transformedFeatures = np.zeros(Xs.shape)
            for i in range(Xs.shape[0]): transformedFeatures[i] = self.perClassTransforms[ys[i]].transfer(Xs[i])
            return transformedFeatures
        else:
            if ys.__class__ is not np.int64: ys = ys.numpy()
            return self.perClassTransforms[ys].transfer(Xs) #Xs_transformed
