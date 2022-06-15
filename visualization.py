#Confusion Matrix  For Multiple Classes

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
dataset = pd.read_csv(r"C:\Users\khza2300\Desktop\chexnet-image\CheXNet-Keras-master\data\default_split\combined_csv")
X = dataset.patientid
y = iris.classes
# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)
knn = KNN(n_neighbors = 3)
# train th model
knn.fit(X_train, y_train)
print('Model is Created')
y_pred = knn.predict(X_test)

y_pred
from sklearn.metrics import confusion_matrix

#Get the confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)

print(cf_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Flower Category')
ax.set_ylabel('Actual Flower Category ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['NoFinding','Atelectasis', 'Cardiomegaly','Effusion','Infiltration','Mass'
                            ,'Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema'
                            ,'Fibrosis','Pleural_Thickening','Hernia'])
ax.yaxis.set_ticklabels(['NoFinding','Atelectasis', 'Cardiomegaly','Effusion','Infiltration','Mass'
                            ,'Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema'
                            ,'Fibrosis','Pleural_Thickening','Hernia'])

## Display the visualization of the Confusion Matrix.
plt.show()


# HISTOGRAM
combined_df.rename({'Finding Labels': 'classes'},
                       axis=1, inplace=True)

#Remove outliers and No finding labels
combined_df["special"] = ""
combined_df["NoFinding"] = ""

combined_df["special"]=(combined_df['classes'].str.contains(r'[@#&$%+-/*|]') )
combined_df["NoFinding"]=(combined_df['classes'].str.contains(r'No Finding') )

combined_df.drop(combined_df[combined_df['special'] == True].index , inplace=True)
combined_df.drop(combined_df[combined_df['NoFinding'] == True].index , inplace=True)

# Plot the most repeated classes

import seaborn as sns

a=combined_df.groupby('classes').count().sort_values(by='Patient ID',ascending=True)

a.head(15)

a.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(25,10))

sns.barplot(x='classes', y='Patient ID', data=a).plot(ax=ax)

ax.set_xlabel("Most Frequented Classes with out Outliers",fontsize=30)

ax.set_ylabel("Frequency",fontsize=30)

ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)

ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18)

ax.bar_label(ax.containers[0])


#AUC Curve





#Acuuracy