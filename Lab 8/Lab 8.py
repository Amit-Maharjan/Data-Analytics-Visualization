import pandas as pd 
dataset = pd.read_csv('Americandata.csv')

from scipy.stats import chi2_contingency
stat, p, dof, expected = chi2_contingency(dataset.iloc[:,1:].values)

print('The chi2 results:\ndof=%d' % dof)
print('p=%s' % p)
print('stat=%s' % stat)
print('Expected Values:\n', expected)

import numpy as np 
from scipy.stats.contingency import margins
def stdres(observed, expected):
    n = observed.sum()
    rsum, csum = margins(observed)
    # Converting into floating point
    rsum = rsum.astype(np.float64)
    csum = csum.astype(np.float64)
    v = csum * rsum * (n - rsum) * (n - csum) / n**3
    return (observed - expected) /np.sqrt(v)

stdResiduals = stdres(dataset.iloc[:,1:].values, expected)
print('\nStandardized Residuals', stdResiduals)