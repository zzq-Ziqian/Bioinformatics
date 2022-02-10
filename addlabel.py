import pandas as pd
import numpy as np

import csv
label=[]
csv_file_credible=list(csv.reader(open('CredibleMeds.csv','r')))
csv_file_offside=list(csv.reader(open('OFFSIDES.csv','r')))

qc=[]
for line1 in csv_file_offside[200002:]:
    qc.append(line1[1])

d1 = list(set(qc))
d1.sort(key=qc.index)
i=0
for line in d1:
    result = 0
    for line2 in csv_file_credible[1:65]:
        if line == line2[0]:
            result = 1
            i+=1
    label.append(result)
print(label)
    #label.append(result)
