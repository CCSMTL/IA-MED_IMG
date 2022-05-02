import os
import shutil


final_folder="../data/"
#training 70 , val 20, test 10 %
n_images=112120
n_train=int(0.7*n_images)
n_val=int(0.2*n_images)
n_test=n_images-n_val-n_train

count=n_images
for folder in os.listdir("../data") :

    while count>0 :
        if count> n_train :
        #add to training set








    count-=1
