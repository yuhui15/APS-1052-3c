"""
In this homework you will learn 
to use a manual cross validation loop that 
focuses on the PROFIT FACTOR as financial scoring function for a regressor. 
Do not use RandomizedSearchCV or GridSearchCV for this purpose.
You will also learn to program Timothy Master's Monte Carlo Permutation Test.

Change this python script file as follows: 
1. instead of using R2 score as the performance metric in the manual cross validation loop,
use the ProfitFactor (see PROFITFACTORFUNCTION.txt)
Compute Profit Factor via a dedicated profit_factor(y_true, y_pred) function.
Save the new python script as ridge_ManualCV_PROFITFACTOR.py

2. starting from ridge_ManualCV_PROFITFACTOR.py 

include new code to implement Timothy Master's full MCPT (Monte Carlo Permutation Test)
discussed in in pp. 291-293 of 
Testing and Tuning Manual Trading Systems. 

Masters MCPT algorithm in pseudo-code is: 

for irep from 0 through nreps-1
      if  (irep > 0)
            shuffle
      for each competitor 
            compute performance of  this competitor for the permuted training targets
            if  (irep == 0)
                  original_performance[competitor] = performance
                  solo_count[competitor] = 1 ;
                  unbiased_count[competitor] = 1 ;
            else
                  if  (performance >= original_performance[competitor])
                        solo_count[competitor] = solo_count[competitor] + 1
      if  (irep > 0)
            best_performance = MAX ( performance of  all competitors )
            for each competitor
                  if  (best_performance >= original_performance[competitor])
                        unbiased_count[competitor] = unbiased_count[competitor] + 1
for all competitors
      solo_pval[competitor] = solo_count[competitor] / nreps
      unbiased_pval[competitor] = unbiased_count[competitor] / nreps

Regarding the above pseudo-code
Use at least 500  nreps.
Use the block permutation function in BLOCKPERMUTATIONFUNCTION.txt
Use at least a BLOCK_SIZE = 5 to guard against target auto-correlation 
(which invalidates the MCPT test).
A "competitor" is a ridge regresor with a particular value of the alpha regularization parameter. 
If you have 10 alpha regularization parameter values to select from, you have 10 competitors.
The original "performance" of a competitor is the ridge regressor's CROSS VALIDATION mean ProfitFactor before permutation.
The "performance" of the competitor is the ridge regressor's CROSS VALIDATION mean ProfitFactor after permutation.
The "shuffle" instruction is to use block permutation on the targets.
Your results should be similar to mcpt_ridgereg_PROFITFACTOR.csv
Save the new python script file as ridge_ManualCV_MonteCarloPermutationTest_PROFITFACTOR.py

What's the point:
The selection rule itself is plain best-CV-score — MCPT doesn't change which competitor wins; MCPT attaches a statistically honest p-value (unbiased_pval) to that winner, answering "is this edge distinguishable from what pure luck produces given a 20-way search?" rather than just "did this one alpha beat chance?"

"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import fAux
import matplotlib.pyplot as plt
import sys

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

np.random.seed(1)  # to fix the results
rs = 2  # kept for consistency with other scripts (not used in manual CV)

# Correct annualization for ~8 bars per trading day (H3)
ANNUALIZATION = 252 * 8  # 2016

# Epsilon added to the Profit Factor denominator to prevent division-by-zero
# blow-up (inf/nan) when a fold has zero losing trades.
EPS_PF = 1e-6

# make sure the output folder for the plots exists
import os
os.makedirs('Results', exist_ok=True)

# file_path = 'outputfile.txt'
# file_path = 'outputfile_ridge_ManualCV_PROFITFACTOR.txt'
file_path = 'outputfile_ridge_MCPT_PROFITFACTOR.txt'
sys.stdout = open(file_path, "w")

# ----------------------------
# MCPT configuration
# ----------------------------
NREPS = 500     # includes irep==0 (unpermuted); at least 500
BLOCK_SIZE = 5  # block length for block permutation of the training target (retFut1)


def block_permutation(x: np.ndarray, block_size: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Block permutation: partition x into contiguous, non-overlapping blocks of
    length block_size (the final block is shorter if len(x) is not an exact
    multiple of block_size), then randomly reorder the blocks as whole units
    and concatenate.

    Used here to permute the training target (retFut1) instead of a plain
    element-wise (i.i.d.) shuffle. Ljung-Box testing on this target shows
    statistically significant serial correlation (p<0.05 at lags 5, 10, 20),
    so a naive element-wise shuffle would destroy that local structure and
    risk generating (predictor, target) pairings that could not occur in
    real life -- Masters' constraint (b) for permuting a target. Block
    permutation keeps blocks of block_size consecutive target values intact,
    preserving short-range dependence, while still randomizing the
    block-to-block ordering (and hence the longer-range link to the
    predictors) that the MCPT is meant to destroy.
    """
    x = np.asarray(x)
    n = len(x)
    n_blocks = int(np.ceil(n / block_size))
    blocks = [x[i * block_size: (i + 1) * block_size] for i in range(n_blocks)]
    block_order = rng.permutation(n_blocks)
    return np.concatenate([blocks[i] for i in block_order])

# --------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------
#df = pd.read_csv('EURUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('GBPUSD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('NZDUSD_H3_200001030000_202107201800.csv', sep='\t')
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')
#df = pd.read_csv('USDCHF_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
# build window momentum features
for n in range(1, 21):
    df[f'ret{n}'] = df["<OPEN>"].pct_change(periods=n).fillna(0)

# build date time features
df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df = df.join(pd.get_dummies(df["hour"], prefix='hour'))
df = df.join(pd.get_dummies(df["day"], prefix='day'))
df.drop(["hour", "day"], axis=1, inplace=True)

# target assuming we know today's open (trade immediately after open)
df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0)

# Since we are trading right after the open, we only know yesterday's OHLCV etc.
df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

# select the features (by dropping)
cols_to_drop = ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>", "<VOL>", "<SPREAD>"]
df_filtered = df.drop(cols_to_drop, axis=1)

# distribute the df data into X inputs and y target
X = df_filtered.drop(['retFut1'], axis=1)
y = df_filtered[['retFut1']]

# select the samples
x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

df_train = df_filtered.iloc[0:10000]
df_test = df_filtered.iloc[10000:12000]

# --------------------------------------------------------------------------------------
# Profit Factor scorer (economic score)
# --------------------------------------------------------------------------------------
def profit_factor(y_true, y_pred):
    """
    y_true: realized future returns (retFut1) on the slice
    y_pred: model predictions (continuous); we trade sign(y_pred)

    PF = sum(profits) / sum(abs(losses))

    An epsilon is added to the denominator to prevent a division-by-zero
    blow-up (inf) when a slice has zero losing trades. When this guard is
    triggered, a printout alerts that the epsilon fallback was applied.
    """
    positions_arr = np.where(y_pred > 0, 1, -1)
    positions_arr = np.nan_to_num(positions_arr, nan=0.0)

    dailyRet_arr = positions_arr * y_true
    dailyRet_arr = np.nan_to_num(dailyRet_arr, nan=0.0)

    profits = np.where(dailyRet_arr >= 0, dailyRet_arr, 0.0)
    losses = np.where(dailyRet_arr < 0, dailyRet_arr, 0.0)

    denom = float(np.sum(np.abs(losses)))
    if denom == 0.0:
        print(
            "[PF guard] Zero losses encountered (sum(profits)={:.6f}); "
            "applying epsilon={} to denominator to prevent blow-up.".format(
                float(np.sum(profits)), EPS_PF
            )
        )
    return float(np.sum(profits)) / (denom + EPS_PF)


# --------------------------------------------------------------------------------------
# Manual CV loop over alpha (TimeSeriesSplit)
# --------------------------------------------------------------------------------------
imputer = SimpleImputer(strategy="constant", fill_value=0)
# Turn off scaling because we should not scale dummies (and returns are already mostly scaled)
scaler = StandardScaler(with_mean=False, with_std=False)
ridge = Ridge(max_iter=1000)

pipe = Pipeline([
    ("imputer", imputer),
    ("scaler", scaler),
    ("ridge", ridge)
])

alpha_grid = np.logspace(-7, 0, num=20, endpoint=True)

# Time-series split (fixed rolling window)
split = TimeSeriesSplit(n_splits=5, max_train_size=2000)

cv_rows = []
best_alpha = None
best_cv_score = -np.inf

X_train_values = x_train.values
y_train_values = y_train.values.ravel()

for alpha in alpha_grid:
    fold_scores = []

    for train_idx, val_idx in split.split(X_train_values):
        X_tr = X_train_values[train_idx]
        y_tr = y_train_values[train_idx]
        X_va = X_train_values[val_idx]
        y_va = y_train_values[val_idx]  # realized returns on validation slice

        pipe.set_params(ridge__alpha=alpha)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)

        # r2_fold = r2_score(y_va, preds)
        # fold_scores.append(r2_fold)
        pf_fold = profit_factor(y_va, preds)  # Profit Factor instead of R2
        fold_scores.append(pf_fold)

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    # cv_rows.append({
    #     "ridge__alpha": alpha,
    #     "mean_cv_r2": mean_score,
    #     "std_cv_r2": std_score
    # })
    cv_rows.append({
        "ridge__alpha": alpha,
        "mean_cv_profit_factor": mean_score,
        "std_cv_profit_factor": std_score
    })

    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_alpha = alpha

print("Best parameters : {}".format({"ridge__alpha": best_alpha}))
# print("Best cross-validation mean R2 : {:.6f}".format(best_cv_score))
print("Best cross-validation mean Profit Factor : {:.6f}".format(best_cv_score))

# results = pd.DataFrame(cv_rows).sort_values("mean_cv_r2", ascending=False)
# results.to_csv("results_ridgereg_r2.csv", index=False)
results = pd.DataFrame(cv_rows).sort_values("mean_cv_profit_factor", ascending=False)
results.to_csv("results_ridgereg_profitfactor.csv", index=False)


# --------------------------------------------------------------------------------------
# Timothy Masters' MCPT (Monte Carlo Permutation Test), pp. 291-293
# A "competitor" = Ridge with one value of alpha (20 competitors).
# Performance of a competitor = CROSS VALIDATION mean Profit Factor.
# --------------------------------------------------------------------------------------
def cv_mean_profit_factor(X_values: np.ndarray, y_values: np.ndarray, alpha: float) -> float:
    """Mean TimeSeriesSplit CV Profit Factor of Ridge(alpha) on (X_values, y_values)."""
    fold_pf = []
    for train_idx, val_idx in split.split(X_values):
        X_tr = X_values[train_idx]
        y_tr = y_values[train_idx]
        X_va = X_values[val_idx]
        y_va = y_values[val_idx]  # (possibly permuted) returns on the validation slice

        pipe.set_params(ridge__alpha=alpha)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        fold_pf.append(profit_factor(y_va, preds))
    return float(np.nanmean(fold_pf))


perm_rng = np.random.RandomState(rs)  # RNG used for the shuffles

original_performance = {}  # alpha -> CV mean PF on unpermuted targets
solo_count = {}            # alpha -> count(perm perf >= original perf of this alpha)
unbiased_count = {}        # alpha -> count(best perm perf of all alphas >= original perf of this alpha)

for irep in range(NREPS):
    if irep > 0:
        # shuffle: block permutation of the training targets
        y_use = block_permutation(y_train_values, BLOCK_SIZE, perm_rng)
    else:
        y_use = y_train_values

    perf_by_alpha = {}
    for alpha in alpha_grid:
        alpha = float(alpha)
        performance = cv_mean_profit_factor(X_train_values, y_use, alpha)
        perf_by_alpha[alpha] = performance

        if irep == 0:
            original_performance[alpha] = performance
            solo_count[alpha] = 1
            unbiased_count[alpha] = 1
        else:
            if performance >= original_performance[alpha]:
                solo_count[alpha] = solo_count[alpha] + 1

    if irep > 0:
        best_performance = max(perf_by_alpha.values())
        for alpha in alpha_grid:
            alpha = float(alpha)
            if best_performance >= original_performance[alpha]:
                unbiased_count[alpha] = unbiased_count[alpha] + 1

mcpt_rows = []
for alpha in alpha_grid:
    alpha = float(alpha)
    mcpt_rows.append({
        "ridge__alpha": alpha,
        "original_cv_profit_factor": original_performance[alpha],
        "solo_count": solo_count[alpha],
        "unbiased_count": unbiased_count[alpha],
        "solo_pval": solo_count[alpha] / NREPS,
        "unbiased_pval": unbiased_count[alpha] / NREPS
    })

mcpt_df = pd.DataFrame(mcpt_rows).sort_values("original_cv_profit_factor", ascending=False)
mcpt_df.to_csv("mcpt_ridgereg_PROFITFACTOR.csv", index=False)

# MCPT does not change the winner (best CV score); it attaches p-values to it
print("MCPT permutation method: block permutation of retFut1 (block_size={}), nreps={}".format(BLOCK_SIZE, NREPS))
print("MCPT solo p-value (best alpha)      : {:.6f}".format(solo_count[float(best_alpha)] / NREPS))
print("MCPT unbiased p-value (best alpha)  : {:.6f}".format(unbiased_count[float(best_alpha)] / NREPS))

# Fit best model on full training set
pipe.set_params(ridge__alpha=best_alpha)
pipe.fit(X_train_values, y_train_values)

# --------------------------------------------------------------------------------------
# Train set performance
# --------------------------------------------------------------------------------------
positions = np.where(pipe.predict(x_train.values) > 0, 1, -1)
dailyRet = positions * df_train.retFut1.values
dailyRet = np.nan_to_num(dailyRet)

cumret = pd.Series(np.cumprod(dailyRet + 1) - 1, index=df_train.index)

plt.figure(1)
plt.plot(cumret.index, cumret.values)
plt.title('Cross-validated RidgeRegression on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.savefig(r'Results\%s.png' % ("TrainCumulative"))

cagr = (1 + cumret.iloc[-1]) ** (ANNUALIZATION / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret.values)
ratio = (ANNUALIZATION ** 0.5) * np.mean(dailyRet) / np.std(dailyRet)

print(
    'In-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'
    .format(cagr, ratio, maxDD, int(maxDDD), -cagr / maxDD if maxDD != 0 else np.nan)
)

# --------------------------------------------------------------------------------------
# Test set performance
# --------------------------------------------------------------------------------------
positions2 = np.where(pipe.predict(x_test.values) > 0, 1, -1)
dailyRet2 = positions2 * df_test.retFut1.values
dailyRet2 = np.nan_to_num(dailyRet2)

cumret2 = pd.Series(np.cumprod(dailyRet2 + 1) - 1, index=df_test.index)

plt.figure(2)
plt.plot(cumret2.index, cumret2.values)
plt.title('Cross-validated RidgeRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.savefig(r'Results\%s.png' % ("TestCumulative"))

rho, pval = spearmanr(y_test.values.ravel(), pipe.predict(x_test.values))  # Spearman rank correlation

cagr2 = (1 + cumret2.iloc[-1]) ** (ANNUALIZATION / len(cumret2)) - 1
maxDD2, maxDDD2 = fAux.calculateMaxDD(cumret2.values)
ratio2 = (ANNUALIZATION ** 0.5) * np.mean(dailyRet2) / np.std(dailyRet2)

print(
    'Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} '
    'maxDDD={:d} Calmar ratio={:0.6}  Rho={:0.6} PVal={:0.6}\n'
    .format(cagr2, ratio2, maxDD2, int(maxDDD2), -cagr2 / maxDD2 if maxDD2 != 0 else np.nan, rho, pval)
)