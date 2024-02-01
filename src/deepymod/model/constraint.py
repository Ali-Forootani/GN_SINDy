"""This module contains concrete implementations of the constraint component."""


import torch
import numpy as np
from .deepmod import Constraint
from typing import List

TensorList = List[torch.Tensor]


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class LeastSquares(Constraint):
    def __init__(self) -> None:
        """Least Squares Constraint solved by QR decomposition"""
        super().__init__()

    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Calculates the coefficients of the constraint using the QR decomposition for every pair
        of sparse feature matrix and time derivative.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples, n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).

        Returns:
            (TensorList): List of calculated coefficients of size [(n_active_features, 1) x n_outputs].
        """
        coeff_vectors = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            Q, R = torch.qr(theta)  # solution of lst. sq. by QR decomp.
            coeff_vectors.append(torch.inverse(R) @ Q.T @ dt)
        return coeff_vectors


class GradParams(Constraint):
    def __init__(self, n_params: int, n_eqs: int) -> None:
        """Constrains the neural network by optimizing over the coefficients together with the network.
           Coefficient vectors are randomly initialized from a standard Gaussian.

        Args:
            n_params (int): number of features in feature matrix.
            n_eqs (int): number of outputs / equations to be discovered.
        """
        super().__init__()
        self.coeffs = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.randn(n_params, 1)) for _ in torch.arange(n_eqs)]
        )

    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList):
        """Returns the coefficients of the constraint, since we're optimizing them by
           gradient descent.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors of size (n_samples, n_active_features).
            time_derivs (TensorList): List containing the time derivatives of size (n_samples, n_outputs).

        Returns:
            (TensorList): Calculated coefficients of size (n_features, n_outputs).
        """
        return self.coeffs


class Ridge(Constraint):
    """Implements the constraint as a least squares problem solved by QR decomposition."""

    def __init__(self, l=1e-3) -> None:
        super().__init__()
        self.l = torch.tensor(l)

    def fit(self, sparse_thetas: TensorList, time_derivs: TensorList) -> TensorList:
        """Calculates the coefficients of the constraint using the QR decomposition for every pair
        of sparse feature matrix and time derivative.

        Args:
            sparse_thetas (TensorList): List containing the sparse feature tensors.
            time_derivs (TensorList): List containing the time derivatives.

        Returns:
            (TensorList): Calculated coefficients.
        """
        coeff_vectors = []
        for theta, dt in zip(sparse_thetas, time_derivs):
            # We use the augmented data method
            norm = torch.norm(theta, dim=0)
            l_normed = torch.diag(
                torch.sqrt(self.l) * norm
            )  # we norm l rather than theta cause we want unnormed coeffs out
            X = torch.cat((theta, l_normed), dim=0)
            y = torch.cat(
                (dt, torch.zeros((theta.shape[1], 1)).to(theta.device)), dim=0
            )
            # Now solve like normal OLS prob lem
            Q, R = torch.qr(X)  # solution of lst. sq. by QR decomp.
            coeff_vectors.append(torch.inverse(R) @ Q.T @ y)

        return coeff_vectors

###############################
###############################


class STRidgeCons(Constraint):
    
    def __init__(self, lam = 0.00000, maxit = 100, tol = 0.05, normalize = 2, print_results = False):
        super().__init__()
        self.lam = lam
        self.tol = tol
        self.normalize = normalize
        self.print_results = print_results
        self.maxit = maxit
        
        
    def fit(self, theta: TensorList, dt: TensorList):
        
        """
        STRidge(TrainR,TrainY,lam,STR_iters,tol,normalize = normalize)
        Sequential Threshold Ridge Regression algorithm for finding (hopefully) sparse 
        approximation to X^{-1}y.  The idea is that this may do better with correlated observables.

        This assumes y is only one column
        """
        
        X0 = theta[0].detach().cpu().numpy()
        y = dt[0].detach().cpu().numpy()
        
        n,d = X0.shape
        
       
        X = np.zeros((n,d))
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
                    #w
                    #print("once this condition was true!")
                    #print(w)
                    
                    return  [torch.tensor(w.reshape(-1,1), dtype=theta[0].dtype ).to(theta[0].device)
                             ]     
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
            
            return [
                torch.tensor(np.multiply(Mreg.reshape(-1,1),
                                             w.reshape(-1,1)),
                             dtype=theta[0].dtype ).to(theta[0].device)
                    ]
        else: 
                        
            return [torch.tensor(w.reshape(-1,1), dtype=theta[0].dtype ).to(theta[0].device)
                    ]





###############################
###############################

class STRidgeCon(Constraint):
    
    def __init__(self, lam = 0.000001, maxit = 100, tol = 0.1, normalize = 2, print_results = False):
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
        
        X0 = X0.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        
        
        
        n,d = X0[0].size()
        
       
        #X = np.zeros((n,d), dtype=np.complex64)
        X = torch.zeros((n,d), dtype = X0[0].dtype).to(X0[0].device)
        
        
        # First normalize data
        if self.normalize != 0:
            Mreg = torch.zeros((d,1)).to(X0[0].device)
            for i in range(0,d):
                Mreg[i] = 1.0/(torch.linalg.norm(X0[0][:,i].clone(),self.normalize))
                X[:,i] = Mreg[i]*X0[0][:,i].clone()
        else: X = X0[0].clone()
        
        
       
        
        
        #################################
        #################################
        
        
        # Get the standard ridge esitmate
        if self.lam != 0:
            
            w = torch.linalg.lstsq(X.t().matmul(X) +
                               self.lam * torch.eye(X.shape[1]).to(device),
                               X.t().matmul(y[0]), rcond=None).solution
            """
            w = torch.linalg.lstsq(X.T @ X + self.lam*torch.eye(d).to(X.device)
                                   , X.T @ y[0],rcond=None)[0].to(X.device)
            """
            
        else:
            
            
            w = torch.linalg.lstsq(X, y[0],rcond=None)[0].to(X.device)
            
        
        num_relevant = d
        biginds = torch.where( abs(w) > self.tol)[0]
        
       
        
        # Threshold and continue
        for j in range(self.maxit):

            # Figure out which items to cut out
            smallinds = torch.where( abs(w) < self.tol)[0]
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
                
                print("jjjjjjjjjjjjjjjj")
                print(X.size())
                print(y[0].size())
                print("jjjjjjjjjjjjjjjj")
                
                """
                w[biginds] = torch.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds])
                                             + self.lam*torch.eye(len(biginds).to(X.device)),
                                             X[:, biginds].T @ y[0],rcond=None).solution
                """
                
                w[biginds] = torch.linalg.lstsq(X[:, biginds],y[0],rcond=None)[0].to(X.device)
                
                
            else: 
                
                w[biginds] = torch.linalg.lstsq(X[:, biginds],y[0],rcond=None)[0].to(X.device)


        # Now that we have the sparsity pattern, use standard least squares to get w
        if biginds != []: 
            
            w[biginds] = torch.linalg.lstsq(X[:, biginds],y[0],rcond=None)[0].to(X.device)
        
        if self.normalize != 0: 
            
            return [torch.multiply(Mreg.reshape(-1,1),w.reshape(-1,1)).to(X.device)]
        
        else: 
                        
            return [w.reshape(-1,1).to(X.device)]



###############################
###############################
"""
class STRidgeCons(Constraint):
    def __init__(
        self, lam=10**-5, maxit=20, tol=0.1, normalize=2, print_results=False
    ):
        super().__init__()
        self.lam = lam
        self.tol = tol
        self.normalize = normalize
        self.print_results = print_results
        self.maxit = maxit

    def fit(self, X0: torch.tensor, y: torch.tensor):
        

        #with torch.no_grad():
        n, d = X0[0].size()

            # n,d = X0.shape

            # First normalize data
        X = torch.zeros((n, d), dtype=torch.float).to(X0[0].device)

            # X.requires_grad = True

        if self.normalize != 0:
            Mreg = torch.zeros((d, 1)).to(X0[0].device)
            for i in range(0, d):
                Mreg[i] = 1.0 / (torch.linalg.norm(X0[0][:, i], self.normalize))
                X[:, i] = Mreg[i] * X0[0][:, i]
            else:
                X = X0[0].detach()

            print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
            print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
            print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
            print(type(X))

            # print(torch.t(X) @ X)

            # Get the standard ridge esitmate
        if self.lam != 0:
            w = torch.linalg.lstsq(
                X.T @ X + self.lam * torch.eye(d).to(X0[0].device),
                X.T @ y[0],
                rcond=None,
            )[0]
            # w.requires_grad = True
        else:
            w = torch.linalg.lstsq(X, y, rcond=None)[0]
            # w.requires_grad = True
        num_relevant = d
        biginds = torch.where(abs(w) > self.tol)[0]

        print("7777777777777777777777777777777777777777")
        print(w)
        print(d)
        print(X.size())
        print(biginds)
        print(num_relevant)
        print("7777777777777777777777777777777777777777")

        # X = np.zeros((n,d), dtype=np.complex64)
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
        
        
        
        # Get the standard ridge esitmate
        if self.lam != 0: 
            w = np.linalg.lstsq(X.T.dot(X) + self.lam*np.eye(d),X.T.dot(y),rcond=None)[0]
        else: 
            w = np.linalg.lstsq(X,y,rcond=None)[0]
            num_relevant = d
            biginds = np.where( abs(w) > self.tol)[0]
        
        # print("uuuuuuuuuuuuuuuuuuuuuuuu")
        # print(w)
        # print("uuuuuuuuuuuuuuuuuuuuuuuu")

        # Threshold and continue
        for j in range(self.maxit):
        # Figure out which items to cut out
            smallinds = torch.where(abs(w) < self.tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]

                # If nothing changes then stop
        if num_relevant == len(new_biginds):
            break
        else:
            num_relevant = len(new_biginds)

            # Also make sure we didn't just lose all the coefficients
        if len(new_biginds) == 0:
               if j == 0:
                    # if print_results: print "Tolerance too high - all coefficients set below tolerance"
                    return w
                else:
                    break
            biginds = new_biginds

            # Otherwise get a new guess
            w[smallinds] = 0

            print("here it is in constraint line 227")
            print(w)

            if self.lam != 0:
                w[biginds] = torch.linalg.lstsq(
                    X[:, biginds].T @ X[:, biginds]
                    + self.lam * torch.eye(len(biginds)).to(X.device),
                    X[:, biginds].T @ y[0],
                    rcond=None,
                )[0]
            else:
                w[biginds] = torch.linalg.lstsq(X[:, biginds], y[0], rcond=None)[0]

            # Now that we have the sparsity pattern, use standard least squares to get w
            if biginds != []:
                w[biginds] = torch.linalg.lstsq(X[:, biginds], y[0], rcond=None)[0]

            if self.normalize != 0:
                return [torch.multiply(Mreg.reshape(-1, 1), w.reshape(-1, 1))]

            else:
                return [w.reshape(-1, 1)]
"""