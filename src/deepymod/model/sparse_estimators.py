"""Sparsity estimators which can be plugged into deepmod.
We keep the API in line with scikit learn (mostly), so scikit learn can also be plugged in.
See scikitlearn.linear_models for applicable estimators."""

import numpy as np
from .deepmod import Estimator
from sklearn.cluster import KMeans
#from pysindy.optimizers import STLSQ
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning
)  # To silence annoying pysindy warnings


class Base(Estimator):
    def __init__(self, estimator: BaseEstimator) -> None:
        """Basic sparse estimator class; simply a wrapper around the supplied sk-learn compatible estimator.

        Args:
            estimator (BaseEstimator): Sci-kit learn estimator.
        """
        super().__init__()
        self.estimator = estimator
        self.estimator.set_params(
            fit_intercept=False
        )  # Library contains offset so turn off the intercept

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns an array with the coefficient verctor after sparsity estimation.

        Args:
            X (np.ndarray): Training input data of shape (n_samples, n_features).
            y (np.ndarray): Training target data of shape (n_samples, n_outputs).

        Returns:
            np.ndarray: Coefficient vector (n_features, n_outputs).
        """
        coeffs = self.estimator.fit(X, y).coef_
        return coeffs


class Threshold(Estimator):
    def __init__(
        self,
        threshold: float = 0.1,
        estimator: BaseEstimator = LassoCV(cv=5, fit_intercept=False),
    ) -> None:
        """Performs additional thresholding on coefficient result from supplied estimator.

        Args:
            threshold (float, optional): Value of the threshold above which the terms are selected. Defaults to 0.1.
            estimator (BaseEstimator, optional): Sparsity estimator. Defaults to LassoCV(cv=5, fit_intercept=False).
        """
        super().__init__()
        self.estimator = estimator
        self.threshold = threshold

        # Library contains offset so turn off the intercept
        self.estimator.set_params(fit_intercept=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns an array with the coefficient verctor after sparsity estimation.

        Args:
            X (np.ndarray): Training input data of shape (n_samples, n_features).
            y (np.ndarray): Training target data of shape (n_samples, n_outputs).

        Returns:
            np.ndarray: Coefficient vector (n_features, n_outputs).
        """
        coeffs = self.estimator.fit(X, y).coef_
        coeffs[np.abs(coeffs) < self.threshold] = 0.0
        
        
       

        return coeffs


class Clustering(Estimator):
    def __init__(
        self, estimator: BaseEstimator = LassoCV(cv=5, fit_intercept=False)
    ) -> None:
        """Performs additional thresholding by Kmeans-clustering on coefficient result from estimator.

        Args:
            estimator (BaseEstimator, optional): Estimator class. Defaults to LassoCV(cv=5, fit_intercept=False).
        """
        super().__init__()
        self.estimator = estimator
        self.kmeans = KMeans(n_clusters=2)

        # Library contains offset so turn off the intercept
        self.estimator.set_params(fit_intercept=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns an array with the coefficient verctor after sparsity estimation.

        Args:
            X (np.ndarray): Training input data of shape (n_samples, n_features).
            y (np.ndarray): Training target data of shape (n_samples, n_outputs).

        Returns:
            np.ndarray: Coefficient vector (n_features, n_outputs).
        """
        coeffs = self.estimator.fit(X, y).coef_[:, None]  # sklearn returns 1D
        clusters = self.kmeans.fit_predict(np.abs(coeffs)).astype(np.bool)

        # make sure terms to keep are 1 and to remove are 0
        max_idx = np.argmax(np.abs(coeffs))
        if clusters[max_idx] != 1:
            clusters = ~clusters

        coeffs = clusters.astype(np.float32)
        return coeffs


##########

class STRidge(Estimator):
    
    def __init__(self, lam = 0.0000, maxit = 100, tol = 0.1, normalize = 2, print_results = False):
        super().__init__()
        self.lam = lam
        self.tol = tol
        self.normalize = normalize
        self.print_results = print_results
        self.maxit = maxit
        
        
    def fit(self, X0: np.ndarray, y: np.ndarray):
        
        """
        STRidge(TrainR,TrainY,lam,STR_iters,tol,normalize = normalize)
        Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
        approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

        This assumes y is only one column
        """
        n,d = X0.shape
       
        X = np.zeros((n,d), dtype=np.complex64)
        # First normalize data
        if self.normalize != 0:
            Mreg = np.zeros((d,1))
            for i in range(0,d):
                Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],self.normalize))
                X[:,i] = Mreg[i]*X0[:,i]
        else: X = X0
        
        # Get the standard ridge esitmate
        if self.lam != 0: 
            w = np.linalg.lstsq(X.T.dot(X) + self.lam*np.eye(d),X.T.dot(y),rcond=None)[0]
        else: 
            w = np.linalg.lstsq(X,y,rcond=None)[0]
        num_relevant = d
        biginds = np.where( abs(w) > self.tol)[0]
        
        # Threshold and continue
        for j in range(self.maxit):

            # Figure out which items to cut out
            smallinds = np.where( abs(w) < self.tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]
                
            # If nothing changes then stop
            if num_relevant == len(new_biginds): break
            else: num_relevant = len(new_biginds)
                
            # Also make sure we didn't just lose all the coefficients
            if len(new_biginds) == 0:
                if j == 0: 
                    #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                    return w
                else: break
            biginds = new_biginds
            
            # Otherwise get a new guess
            w[smallinds] = 0
            
            if self.lam != 0: 
                
                w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + self.lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=None)[0]
            else: 
                
                w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]

        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []: 
            
            w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
        
        if self.normalize != 0: 
            
            return np.multiply(Mreg.reshape(-1,1),w.reshape(-1,1))
        
        else: 
                        
            return w.reshape(-1,1)
        
        
    
def STRidge_func(X0, y, lam, maxit, tol, normalize = 2, print_results = False):
    """
    STRidge(TrainR,TrainY,lam,STR_iters,tol,normalize = normalize)
    Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
    approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

    This assumes y is only one column
    """

    n,d = X0.shape
    X = np.zeros((n,d), dtype=np.complex64)
    # First normalize data
    if normalize != 0:
        Mreg = np.zeros((d,1))
        for i in range(0,d):
            Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
            X[:,i] = Mreg[i]*X0[:,i]
    else: X = X0
    
    # Get the standard ridge esitmate
    if lam != 0: w = np.linalg.lstsq(X.T.dot(X) + lam*np.eye(d),X.T.dot(y),rcond=None)[0]
    else: w = np.linalg.lstsq(X,y,rcond=None)[0]
    num_relevant = d
    biginds = np.where( abs(w) > tol)[0]
    
    # Threshold and continue
    for j in range(maxit):

        # Figure out which items to cut out
        smallinds = np.where( abs(w) < tol)[0]
        new_biginds = [i for i in range(d) if i not in smallinds]
            
        # If nothing changes then stop
        if num_relevant == len(new_biginds): break
        else: num_relevant = len(new_biginds)
            
        # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
            if j == 0: 
                #if print_results: print "Tolerance too high - all coefficients set below tolerance"
                return w
            else: break
        biginds = new_biginds
        
        # Otherwise get a new guess
        w[smallinds] = 0
        if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y),rcond=None)[0]
        else: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]

    # Now that we have the sparsity pattern, use standard least squares to get w
    if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y,rcond=None)[0]
    
    if normalize != 0: return np.multiply(Mreg,w)
    else: return w




















#########





class PDEFIND(Estimator):
    def __init__(self, lam: float = 1e-3, dtol: float = 0.1) -> None:
        
        super().__init__()
        self.lam = lam
        self.dtol = dtol

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
       
        coeffs = PDEFIND.TrainSTLSQ(X, y[:, None], self.lam, self.dtol)
        return coeffs.squeeze()

    @staticmethod
    def TrainSTLSQ(
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        delta_threshold: float,
        max_iterations: int = 100,
        test_size: float = 0.2,
        random_state: int = 0,
    ) -> np.ndarray:
       
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Set up the initial tolerance l0 penalty and estimates
        l0 = 1e-3 * np.linalg.cond(X)
        delta_t = delta_threshold  # for interal use, can be updated

        # Initial estimate
        optimizer = STLSQ(
            threshold=0, alpha=0.0, fit_intercept=False
        )  # Now similar to LSTSQ
        y_predict = optimizer.fit(X_train, y_train).predict(X_test)
        min_loss = np.linalg.norm(y_predict - y_test, 2) + l0 * np.count_nonzero(
            optimizer.coef_
        )

        # Setting alpha and tolerance
        best_threshold = delta_t
        threshold = delta_t

        for iteration in np.arange(max_iterations):
            optimizer.set_params(alpha=alpha, threshold=threshold)
            y_predict = optimizer.fit(X_train, y_train).predict(X_test)
            loss = np.linalg.norm(y_predict - y_test, 2) + l0 * np.count_nonzero(
                optimizer.coef_
            )

            if (loss <= min_loss) and not (np.all(optimizer.coef_ == 0)):
                min_loss = loss
                best_threshold = threshold
                threshold += delta_threshold

            else:  # if loss increases, we need to a) lower the current threshold and/or decrease step size
                new_lower_threshold = np.max([0, threshold - 2 * delta_t])
                delta_t = 2 * delta_t / (max_iterations - iteration)
                threshold = new_lower_threshold + delta_t

        optimizer.set_params(alpha=alpha, threshold=best_threshold)
        optimizer.fit(X_train, y_train)

        return optimizer.coef_

