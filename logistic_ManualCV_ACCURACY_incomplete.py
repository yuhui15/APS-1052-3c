
"""
In this homework you will learn 
to use a manual cross valudation loop that 
focuses on the PROFIT FACTOR as financial scoring function for a classifier. 
Do not use RandomizedSearchCV or GridSearchCV for this purpose.
You will also learn to program Timothy Master's Monte Carlo Permutation Test.

Change this python script file as follows: 
1. instead of using accuracy as the performance metric in the manual cross validation loop,
use the ProfitFactor (see PROFITFACTORFUNCTION.txt)
Compute Profit Factor via a dedicated profit_factor(y_true, y_pred) function.
Save the new python script as logistic_ManualCV_PROFITFACTOR.py

2. starting from logistic_ManualCV_PROFITFACTOR.py 

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
A "competitor" is a logistic regresor with a particular value of the C regularization parameter. 
If you have 10 C regularization parameter values to select from, you have 10 competitors.
The original "performance" of a competitor is the logistic regressor's CROSS VALIDATION mean ProfitFactor before permutation.
The "performance" of the competitor is the logistic regressor's CROSS VALIDATION mean ProfitFactor after permutation.
The "shuffle" instruction is to use block permutation on the targets.

Your results should be similar to mcpt_logisticreg_PROFITFACTOR.csv
Save the new python script file as logistic_ManualCV_MonteCarloPermutationTest_PROFITFACTOR.py

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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

np.random.seed(1)  # to fix the results
rs = 2

# Correct annualization for ~8 bars per trading day (H3)
ANNUALIZATION = 252 * 8  # 2016

file_path = 'outputfile.txt'
sys.stdout = open(file_path, "w")

# --------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------
df = pd.read_csv('USDCAD_H3_200001030000_202107201800.csv', sep='\t')

df['<DATETIME>'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'])
df = df.set_index('<DATETIME>')
df.drop(['<TIME>'], axis=1, inplace=True)
df.drop(['<DATE>'], axis=1, inplace=True)

# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
for n in range(1, 21):
    df[f'ret{n}'] = df["<OPEN>"].pct_change(periods=n).fillna(0)

df["hour"] = df.index.hour.values
df["day"] = df.index.dayofweek.values
df = df.join(pd.get_dummies(df["hour"], prefix='hour'))
df = df.join(pd.get_dummies(df["day"], prefix='day'))

df['retFut1'] = df['<OPEN>'].pct_change(1).shift(-1).fillna(0)
df['retFut1_categ'] = np.where(df['retFut1'] > 0, 1, 0)

df['<HIGH>'] = df['<HIGH>'].shift(1)
df['<LOW>'] = df['<LOW>'].shift(1)
df['<CLOSE>'] = df['<CLOSE>'].shift(1)
df['<VOL>'] = df['<VOL>'].shift(1)
df['<SPREAD>'] = df['<SPREAD>'].shift(1)

cols_to_drop = ["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>", "<VOL>", "<SPREAD>"]
df_filtered = df.drop(cols_to_drop, axis=1)

X = df_filtered.drop(['retFut1', 'retFut1_categ', "hour", "day"], axis=1)
y = df_filtered['retFut1_categ'].astype(int)

x_train = X.iloc[0:10000]
x_test = X.iloc[10000:12000]

y_train = y.iloc[0:10000]
y_test = y.iloc[10000:12000]

df_train = df_filtered.iloc[0:10000]
df_test = df_filtered.iloc[10000:12000]

# --------------------------------------------------------------------------------------
# Manual grid search over C (TimeSeries CV)
# --------------------------------------------------------------------------------------
imputer = SimpleImputer(strategy="constant", fill_value=0)
scaler = StandardScaler(with_mean=False, with_std=False)

base_logistic = LogisticRegression(
    max_iter=1000,
    solver='liblinear',
    penalty='l2'
)

pipe = Pipeline([
    ("imputer", imputer),
    ("scaler", scaler),
    ("logistic", base_logistic)
])

c_grid = np.logspace(3, -4, num=20, endpoint=True)

# 🔁 Time-series split (fixed rolling window)
split = TimeSeriesSplit(n_splits=5, max_train_size=2000)

cv_rows = []
best_c = None
best_cv_score = -np.inf

for C in c_grid:
    fold_scores = []

    for train_idx, val_idx in split.split(x_train.values):
        X_tr = x_train.values[train_idx]
        y_tr = y_train.values[train_idx]
        X_va = x_train.values[val_idx]
        y_va = y_train.values[val_idx]

        pipe.set_params(logistic__C=C)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        fold_scores.append(accuracy_score(y_va, preds))

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    cv_rows.append({
        "C": C,
        "mean_cv_accuracy": mean_score,
        "std_cv_accuracy": std_score
    })

    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_c = C

print(f"Best parameters : {{'logistic__C': {best_c}, 'logistic__penalty': 'l2'}}")
print("Best cross-validation score : {:.2f}".format(best_cv_score * 100))

results = pd.DataFrame(cv_rows).sort_values("mean_cv_accuracy", ascending=False)
results.to_csv("results_logisticreg.csv", index=False)

# Fit best model on full training set
pipe.set_params(logistic__C=best_c)
pipe.fit(x_train.values, y_train.values)

# --------------------------------------------------------------------------------------
# Train set performance
# --------------------------------------------------------------------------------------
positions = np.where(pipe.predict(x_train.values) > 0, 1, -1)
dailyRet = positions * df_train.retFut1.values
dailyRet = np.nan_to_num(dailyRet)

cumret = np.cumprod(dailyRet + 1) - 1

plt.figure(1)
plt.plot(df_train.index, cumret)
plt.title('Cross-validated LogisticRegression on currency: train set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.savefig(r'Results\%s.png' % ("TrainCumulative"))

cagr = (1 + cumret[-1]) ** (ANNUALIZATION / len(cumret)) - 1
maxDD, maxDDD = fAux.calculateMaxDD(cumret)
ratio = (ANNUALIZATION ** 0.5) * np.mean(dailyRet) / np.std(dailyRet)

print(
    'In-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} maxDDD={:d} Calmar ratio={:0.6}\n'
    .format(cagr, ratio, maxDD, maxDDD.astype(int), -cagr / maxDD)
)

# --------------------------------------------------------------------------------------
# Test set performance
# --------------------------------------------------------------------------------------
positions2 = np.where(pipe.predict(x_test.values) > 0, 1, -1)
dailyRet2 = positions2 * df_test.retFut1.values
dailyRet2 = np.nan_to_num(dailyRet2)

cumret2 = np.cumprod(dailyRet2 + 1) - 1

plt.figure(2)
plt.plot(df_test.index, cumret2)
plt.title('Cross-validated LogisticRegression on currency: test set')
plt.ylabel('Cumulative Returns')
plt.xlabel('Date')
plt.savefig(r'Results\%s.png' % ("TestCumulative"))

test_accuracy = accuracy_score(y_test.values, pipe.predict(x_test.values))

cagr2 = (1 + cumret2[-1]) ** (ANNUALIZATION / len(cumret2)) - 1
maxDD2, maxDDD2 = fAux.calculateMaxDD(cumret2)
ratio2 = (ANNUALIZATION ** 0.5) * np.mean(dailyRet2) / np.std(dailyRet2)

print(
    'Out-of-sample: CAGR={:0.6} Sharpe ratio={:0.6} maxDD={:0.6} '
    'maxDDD={:d} Calmar ratio={:0.6} accuracy_score={:0.6}\n'
    .format(cagr2, ratio2, maxDD2, maxDDD2.astype(int), -cagr2 / maxDD2, test_accuracy)
)
