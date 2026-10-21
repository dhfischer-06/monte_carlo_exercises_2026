import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats

class DeltaGamma:
    def __init__(self, delta: np.ndarray, gamma: np.ndarray, cov: np.ndarray):
        self.delta = delta
        self.gamma = gamma
        self.cov = cov
        self.m = np.size(cov, 0)        
        self.S_dist = stats.norm

    def loss(self, n_sim: int = 10000) -> np.ndarray:
        """
        Calculate the loss of a portfolio using the Delta-Gamma approximation.

        Parameters
        ----------
        n_sim : int, optional
            The number of simulations to run. Default is 10000.

        Returns
        -------
        np.ndarray
            The estimated losses of the portfolio.
        """
        # Generate random samples from the specified distribution
        Z = self.S_dist.rvs(size=(self.m, n_sim), scale=np.sqrt(np.diag(self.cov)))
        
        # Calculate the change in stock prices
        dS = np.array(Z)
        
        # Calculate the portfolio loss using Delta-Gamma approximation
        loss = self.delta.T @ dS + dS.T @ self.gamma @ dS
        
        return loss

    def linear_loss(self, n_sim: int = 10000) -> np.ndarray:
        """
        Calculate the linear loss of a portfolio using the Delta approximation.

        Parameters
        ----------
        n_sim : int, optional
            The number of simulations to run. Default is 10000.

        Returns
        -------
        np.ndarray
            The estimated linear losses of the portfolio.
        """
        # Generate random samples from the specified distribution
        Z = self.S_dist.rvs(size=(self.m, n_sim), scale=np.sqrt(np.diag(self.cov)))
        
        # Calculate the change in stock prices
        dS = np.array(Z)
        
        # Calculate the portfolio loss using Delta approximation
        linear_loss = self.delta.T @ dS
        
        return linear_loss

def main():
    delta = np.array([[2]])
    gamma = np.array([[1]])
    cov = np.array([[1]])
    n_sim = 10000
    delta_gamma = DeltaGamma(delta, gamma, cov)

    ##### 1. Crude Monte Carlo simulation #####

    # Calculate losses
    losses = delta_gamma.loss(n_sim)
    # P(L>10)
    prob = np.mean(losses > 10)
    # variance of estimation treating the loss as a Bernoulli random variable
    var_estimation = prob * (1 - prob) / n_sim

    print("\n-- Crude Monte Carlo --")
    print("Probability of loss exceeding 10:", prob)
    print("Variance of estimation:", var_estimation)

    ##### 2. Control Variates #####

    # The linear approximation of the loss can be used as 
    # a control variate to reduce the variance of the estimation.
    linear_losses = delta_gamma.linear_loss(n_sim)
    """
    The expected value of the linear loss can be calculated analytically as:
    E[linear_loss] = delta.T @ E[dS] = 0, since E[dS] = 0 for a zero-mean normal distribution.

    The variance of the linear loss can also be calculated analytically as:
    Var[linear_loss] = delta.T @ Cov[dS] @ delta = delta.T @ cov @ delta, which is a scalar value.
    """
    linear_loss_mean = 0
    linear_loss_var = delta.T @ cov @ delta
    # Calculate the covariance between the loss and the linear loss
    cov_control = np.cov(losses, linear_losses)[0, 1]
    # Calculate the optimal coefficient for the control variate
    c = cov_control / linear_loss_var
    # Calculate the control variate-adjusted loss
    controlled_losses = losses - c * (linear_losses - linear_loss_mean)

    # Calculate the estimated probability and variance of the control variate-adjusted loss
    controlled_prob = np.mean(controlled_losses > 10)
    # variance of estimation treating the loss as a Bernoulli random variable
    controlled_var_estimation = controlled_prob * (1 - controlled_prob) / n_sim

    print("\n-- Control Variates --")
    print("Probability of loss exceeding 10:", controlled_prob)
    print("Variance of estimation:", controlled_var_estimation)

if __name__ == "__main__":
    main()