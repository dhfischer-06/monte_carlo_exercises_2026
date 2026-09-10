import argparse
from cProfile import label
from matplotlib import style
import numpy as np
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt

# The sum of n iid variables has
# expectation n * mu, variance n * sigma^2
# i.e., statistics carry over linearly

def main():
    # 1. Initialize the parser with a helpful description
    parser = argparse.ArgumentParser(
        description="lognorm sum exercize."
    )

    # 2. Add an OPTIONAL argument that takes an integer with a default value
    parser.add_argument(
        "-n",
        "--sample_size",
        type=int,
        default=50,
        help="Number of times to run the operation (default: 50)",
    )
    parser.add_argument(
        "-s",
        "--scale",
        type=float,
        default=0.5,
        help="Scale parameter for lognormal distribution (def: 0.5)",
    )

    # 5. Parse the command-line inputs
    args = parser.parse_args()

    num_trials = args.sample_size
    scale = args.scale

    lnorm = stats.lognorm(s=scale)

    Y = 3*lnorm.mean() # theoretical mean
    sigma2 = 3*lnorm.var() # theor. var.
    critical_n = np.ceil(1.96**2 * sigma2 / 0.0001) # theoretical n for eps <= 0.01

    def lnorm_sum(n=num_trials):
        Y_hat = np.sum(lnorm.rvs([n,3]), axis=1) # sample data
        sigma2_hat = np.sum((Y_hat.mean() - Y_hat)**2) / (n - 1) # sample var
        se_hat = np.sqrt(sigma2_hat / n) # sample se
        ci = (Y_hat.mean() - 1.96 * se_hat, Y_hat.mean() + 1.96 * se_hat) # 95% conf. inter.

        critical_n_hat = np.ceil(1.96**2 * sigma2_hat / 0.0001) # calculated n for eps <= 0.01

        return Y_hat, sigma2_hat, se_hat, ci, critical_n_hat

    Y_hat, sigma2_hat, se_hat, ci, critical_n_hat = lnorm_sum()
    
    print(f"\nTheoretical Mean:        {np.round(Y, 4)}")
    print(f"Sample Mean:             {np.round(Y_hat.mean(), 4)}\n")

    print(f"Theoretical Variance:    {np.round(sigma2, 4)}")
    print(f"Sample Variance:         {np.round(sigma2_hat, 4)}\n")

    print(f"Absolute Precision:      {np.round(1.96 * se_hat, 4)}")
    print(f"95% Confidence Interval: {np.round(ci, 4)}\n")

    print(f"Theoretical critical n:  {int(critical_n)}")
    print(f"Estimated critical n:    {int(critical_n_hat)}\n")

    plt.hist(Y_hat, density=True)
    plt.axvline(Y_hat.mean(), c="red")
    plt.axvline(x=ci[0],c='black')
    plt.axvline(x=ci[1],c='black')
    plt.show()

    var_hat_elbow = np.zeros(1000)
    se_hat_elbow = np.zeros(1000)
    x = range(50,50050,50)

    for i, trial in tqdm(enumerate(x)):
        _, var_hat_elbow[i], se_hat_elbow[i], _, _ = lnorm_sum(trial)

    print()

    plt.scatter(x, var_hat_elbow, s=0.7)
    plt.axhline(sigma2, c='red')
    plt.show()

    plt.scatter(x, se_hat_elbow, s=0.7)
    plt.show()

    # crit n scoring

    se_hat_hist = np.zeros(1000)

    for i in tqdm(range(1000)):
        _, _, se_hat_hist[i], _, _ = lnorm_sum(42031)

    abs_error_hist = se_hat_hist * 1.96
    pct_in_precision = 1 - np.sum(np.ceil(abs_error_hist - 0.01)) / 1000
    sigma2_hat_crit_n = np.sum((abs_error_hist.mean() - abs_error_hist)**2) / 999

    print(f"\nPercent within 0.01 precision: {pct_in_precision}")
    print(f"Mean precision:                {abs_error_hist.mean()}")

    plt.hist(abs_error_hist, density=True)
    plt.axvline(abs_error_hist.mean(), c='red', ls='--')
    plt.axvline(0.01, c='black')
    plt.show()


if __name__ == "__main__":
    main()