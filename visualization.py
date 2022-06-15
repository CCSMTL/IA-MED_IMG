#Confusion Matrix  For Multiple Classes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
combined_df=pd.read_csv("combined_csv.csv")


combined_df.rename({'Finding Labels': 'classes'},
                       axis=1, inplace=True)

#print(combined_df["classes"])
combined_df["special"] = ""
combined_df["NoFinding"] = ""

combined_df["special"]=(combined_df['classes'].str.contains(r'[@#&$%+-/*|]') )
combined_df["NoFinding"]=(combined_df['classes'].str.contains(r'No Finding') )

combined_df.drop(combined_df[combined_df['special'] == True].index , inplace=True)
combined_df.drop(combined_df[combined_df['NoFinding'] == True].index , inplace=True)

print(combined_df.shape)
X, y = combined_df[['Patient ID', 'special']], combined_df['classes']
print(X.shape, y.shape)


# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
knn = KNN(n_neighbors = 3)
# train th model

knn.fit(X_train, y_train)
print('Model is Created')
y_pred = knn.predict(X_test)

print(y_pred)
from sklearn.metrics import confusion_matrix

#Get the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

print(cf_matrix)
import seaborn as sns
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(25,10))
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Disease Category')
ax.set_ylabel('Actual Disease Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Atelectasis', 'Cardiomegaly','Effusion','Infiltration','Mass'
                            ,'Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema'
                            ,'Fibrosis','Pleural_Thickening','Hernia'],rotation = 30)
ax.yaxis.set_ticklabels(['Atelectasis', 'Cardiomegaly','Effusion','Infiltration','Mass'
                            ,'Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema'
                            ,'Fibrosis','Pleural_Thickening','Hernia'],rotation = 30)

## Display the visualization of the Confusion Matrix.
plt.savefig('Confusion Matrix.png')

