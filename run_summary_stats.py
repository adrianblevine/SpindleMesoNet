import os
import sys
import pandas as pd
import statistics

verbose = False

csvfile = sys.argv[1]


df = pd.read_csv(csvfile, sep='\t')

if verbose:
  print('rows: {}'.format(len(df)))
  print(df.columns)

aucs = list(df.auc)

mean = round(statistics.mean(aucs), 3)
std = round(statistics.stdev(aucs), 3)

print('\nmean AUC: {} (std dev: {})'.format(mean, std))
