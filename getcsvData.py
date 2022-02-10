import pandas as pd
import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse import csc_matrix



import csv
content=[]
csv_file=list(csv.reader(open('OFFSIDES.csv','r')))
#txt_file=pd.read_excel (r'D:/translational bioinformatics/CredibleMedsDATA.xlsx') # r' local directory + file.name.xlsx '


#compare_list= pd.DataFrame(txt_file, columns= ['names'])
#label_dict = {} # a dict as reference, key=drug name, val=boolean
#print(label_dict)


for line in csv_file[200002:]:
    content.append([line[1],line[3],line[10]])
qc1 = []
for line in content:
    qc1.append(line[0])
d1=list(set(qc1))
d1.sort(key=qc1.index)
qc2 = []
for line in content:
    qc2.append(line[1])
d2=list(set(qc2))
d2.sort(key=qc2.index)

for line in content:
    line[0] = d1.index(line[0])
    line[1] = d2.index(line[1])

row_drug = []
for line in content:
    row_drug.append(line[0])

col_event = []
for line in content:
    col_event.append(line[1])

data_frequency = []
for line in content:
    data_frequency.append(line[2])

offside_row = np.array(row_drug)
offside_col = np.array(col_event)
offside_data = np.array(data_frequency)
offside_coo_matrix = coo_matrix((offside_data,(offside_row,offside_col)),shape=(2730,14544),dtype=np.float)
#offside_csc_matrix = offside_coo_matrix.tocsc()
#print(offside_csc_matrix.toarray())
scipy.sparse.save_npz('D:/translational bioinformatics/offside_coo_matrix.npz',offside_coo_matrix) #储存
#x_train_sparse = sparse.load_npz('path.npz') #读取







#先遍历去重，index

#list_drugname=list.col_drugname

#list_drugname.sort(key=col_drugname.index)
#print(list_drugname)


##def print_distinct(data_1):
  #  items = ["drug_concept_name"]
   # for i in range(len(items)):
   #     s = set()
    #    for line in data_1:





#coo = coo_matrix((col_3, (col_1, col_2)))
#print(coo)
# [[15624510        1]
#  [15810944        1]
#  [15668575        2]
#  [15603246        2]
#  [  ...          ..]]
