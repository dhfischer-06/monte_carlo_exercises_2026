import numpy as np
from scipy.fftpack import cs_diff
import scipy.stats as stat
import matplotlib.pyplot as plt
from tqdm import tqdm

n_strata = 6
N = 10_000
crude_mc_var_single_draw = 5.8 # use properties of summing random variables
excl_0 = False

## Stratification
Z = stat.poisson
E = stat.expon
U = stat.uniform
stratum = np.array(range(n_strata))
p_stratum = Z.pmf(mu=2.9, k=stratum)

print(f"\nPortion of cases covered by strata 0-5:", sum(p_stratum), f"\n")

draws = np.zeros(n_strata)
variances = np.zeros(n_strata)

for strata in stratum:
    if excl_0 and strata == 0:
        continue # draws and variances already 0 in this column
    sim = (np.sum(E.rvs(size=(N, strata)),axis=1) <= 1) # draws strata rv's n times
    draws[strata] = np.mean(sim)
    variances[strata] = np.var(sim, ddof=1)
    
cdf_cutoff = Z.cdf(mu=2.9, k=5)
final_draw_q = U.rvs(loc=cdf_cutoff, scale=1-cdf_cutoff, size=N)
final_draw = Z.ppf(mu=2.9, q=final_draw_q)

final_sum = np.zeros(N)
for i, n_draws in enumerate(tqdm(final_draw)):
    final_sum[i] = np.sum(E.rvs(size=int(n_draws)))
draws = np.append(draws, np.mean(final_sum <= 1))
variances = np.append(variances, np.var((final_sum <= 1), ddof=1))
p_stratum = np.append(p_stratum, 1-cdf_cutoff)

print(f"Average value of draw for each strata:\n",draws,f"\n")

X = np.dot(p_stratum, draws)
var_X = np.sum(p_stratum ** 2 * variances / N)

print(f"Estimated value for X:", X,f"\n")
print(f"Estimated variance for X:", var_X, f"\n")
if excl_0:
    print(f"Estimated crude variance for X:", X * (1-X) / (6*N), f"\n")
else:
    print(f"Estimated crude variance for X:", X * (1-X) / (7*N), f"\n")
