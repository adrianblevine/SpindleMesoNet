import numpy as np

logfile = '/home/alevine/tcga_glioma/molecular/runs_1p19q/6-6-2019/logfile.txt'

with open(logfile) as f:
  all_lines = f.readlines()

lines = [x.split(' ')[4] for x in all_lines if x[:19] == 'poor quality tissue']

slides, counts = [list(x) for x in np.unique(np.array(lines), return_counts=True)]

ordered = sorted(zip(counts, slides), reverse=True)

for x in range(len(ordered)): print(ordered[x])
