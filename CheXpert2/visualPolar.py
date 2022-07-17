import numpy as np
import matplotlib.pyplot as plt


Pathology = ['Cardiomegaly', 'Emphysema', 'Edema', 'Hernia', 'Pneumothorax', 'Effusion', 'Mass', 'Fibrosis',
             'Atelectasis','Consolidation','Pleural Thicken','Nodule','Pneumonia','Infiltration','']

ResNet_50 = [0.810,0.833,0.805,0.872,0.799,0.759,0.693,0.786,0.700,0.703,0.684,0.669,0.658,0.661,0]
ResNet = [0.856,0.842,0.806,0.775,0.805,0.806,0.777,0.743,0.733,0.711,0.724,0.724,0.684,0.673,0]
DenseNet_121 = [0.883,0.895,0.835,0.896,0.846,0.828,0.821,0.818,0.767,0.745,0.761,0.758,0.731,0.709,0]
ResNet_38=[0.875,0.895,0.846,0.937,0.840,0.822,0.820,0.816,0.763,0.749,0.763,0.747,0.714,0.694,0]
Sequentional_Model= [0.953,0.986,0.981,0.997,0.924,0.901,0.901,0.926,0.937,0.845, 0.947, 0.958,0.921,0.916,0.687]


label_loc = np.linspace(start=0   , stop=2 * np.pi, num=len(DenseNet_121))

plt.figure(figsize=(10, 10))
plt.subplot(polar=True)
plt.plot(label_loc, ResNet_50, label='ResNet_50',color='orange')
plt.plot(label_loc, ResNet, label='ResNet',color='green')
plt.plot(label_loc, DenseNet_121, label='DenseNet_121',color='red')
plt.plot(label_loc, ResNet_38, label='ResNet_38',color='blue')
plt.plot(label_loc, Sequentional_Model, label='Sequentional Model')
plt.title('Classification Method Comparison', size=20)
lines, labels = plt.thetagrids(np.degrees(label_loc), labels=Pathology)
plt.legend()

plt.savefig('C:/Users/khza2300/Desktop/chexnet-image/CheXNet-Keras-master/presentation/Classification Method Comparison.jpeg')

plt.show()
