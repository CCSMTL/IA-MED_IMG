#Confusion Matrix  For Multiple Classes


import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
def visualization(results):
   fig, ax = plt.subplots(figsize=(25,10))
   cf_matrix = confusion_matrix(results[0], results[1])
   ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True,
            fmt='.2%', cmap='Blues')

    ax.set_title('Seaborn Confusion Matrix with labels\n\n')
    ax.set_xlabel('\nPredicted Disease Category')
    ax.set_ylabel('Actual Disease Category ')

## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Atelectasis', 'Cardiomegaly','Effusion','Infiltration','Mass'
                            ,'Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema'
                            ,'Fibrosis','Pleural_Thickening','Hernia'],rotation = 30)
    ax.yaxis.set_ticklabels(['Atelectasis', 'Cardiomegaly','Effusion','Infiltration','Mass'
                            ,'Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema'
                            ,'Fibrosis','Pleural_Thickening','Hernia'],rotation = 30)

## Display the visualization of the Confusion Matrix.
    plt.savefig('Confusion Matrix.png')
    return results


if __name__=="__main__" :
    results=torch.zeros((2,100,14))
    results[0]=torch.rand((100,14)).round()
    results[1] = torch.rand((100, 14))
    visualization(results)