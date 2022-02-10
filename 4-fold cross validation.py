import matplotlib.pylab as plt
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve,auc
from sklearn.model_selection import StratifiedKFold
import matplotlib.patches as patches
import numpy as np
import pandas as pd


import scipy
from scipy.sparse import coo_matrix
offside_coo_matrix = scipy.sparse.load_npz(r'D:\translational bioinformatics\offside_coo_matrix.npz')

import csv
label=[]
csv_file_label=list(csv.reader(open('D:/translational bioinformatics/DDI createMatrix/label.csv','r')))
for label_line in csv_file_label:
    label.append(label_line)

x = offside_coo_matrix
y = np.array(label)

clf = RandomForestClassifier(n_estimators=30)
cv = StratifiedKFold(n_splits=4)

fig1 = plt.figure(figsize=(12, 12))
tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)
i = 1
for train,test in cv.split(x,y):
    prediction = clf.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
    fpr, tpr, t = roc_curve(y[test], prediction[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i = i + 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.show()


'''
def plot_roc_cur(fper, tper):
    plt.plot(fper, tper, color='orange', label='rf-ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

rfclf = RandomForestClassifier(n_estimators=30)
rf_scores = cross_val_score(rfclf, x,y, cv=4)
print('Random Forest Cross Validation accuracy scores: %s' % rf_scores)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
rfclf = RandomForestClassifier(n_estimators=30)
rfclf.fit(x_train, y_train)
y_predict = rfclf.predict(x_test)
probs = rfclf.predict_proba(x_test)
probs = probs[:, 1]
fper, tper, thresholds = roc_curve(y_test, probs)
plot_roc_cur(fper, tper)
print('Auc=',auc(fper,tper))
'''