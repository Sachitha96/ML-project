import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  seaborn as sns

data = pd.read_csv('creditcard1.csv')

data = data.sample(frac=0.1, random_state=1)

data.hist(figsize=(20, 20))

fraud=data [data['Class']==1]
valid=data[data['Class']==0]

outlier_fraction = len(fraud)/float(len(valid))
print(outlier_fraction)

print('fraud cases: {} '.format(len(fraud)))
print('valid cases: {} '.format(len(valid)))

corrmat = data.corr()
fig=plt.figure(figsize=(12,9))

sns.heatmap(corrmat, vmax=8, square=True)

columns = data.columns.tolist()

columns = [c for c in columns if c not in["Class"]]

target ="Class"
x =data[columns]
y =data[target]

print(x.shape)
print(y.shape)


from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

state =1
classifiers ={
    "Isolation forest": IsolationForest(max_sample=len(x),
                                        contamination = outlier_fraction,
                                        random_state= state),
    "Local_outlier_factor": LocalOutlierFactor(
        n_neighbors= 20,
        contamination= outlier_fraction)
}

import pickle

model_file = "model.pkl"

n_outliers = len(fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
    clf.fit(x)
    with open(model_file, 'wb') as file:
        pickle.dump(clf,file)
    s = clf.decision_function(x)
    y_pred = clf.predict(x)


    print(len(y_pred))
    print(np.array(y.pred))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))

