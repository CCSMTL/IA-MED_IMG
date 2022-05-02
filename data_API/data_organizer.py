import os
import shutil


final_folder="../data/"
#training 70 , val 20, test 10 %
n_images=112120
n_train=int(0.7*n_images)
n_val=int(0.2*n_images)
n_test=n_images-n_val-n_train

count=n_images
os.mkdir(final_folder+"training",parents=True,exist_ok=True)
os.mkdir(final_folder+"validation",parents=True,exist_ok=True)
os.mkdir(final_folder+"test",parents=True,exist_ok=True)
for folder in os.listdir("../data") :

    for root, dirs, files in os.walk("../data", topdown=False):  #adjust walk parameters
        for file in files :
            if count> n_train :
                #add to training set
                shutil.copy(f"{root}/{dirs}/{file}",f"{root}/training/images/{file}")

            elif count>n_val :
                shutil.copy(f"{root}/{dirs}/{file}", f"{root}/validation/images/{file}")

            else :
                shutil.copy(f"{root}/{dirs}/{file}", f"{root}/test/images/{file}")








    count-=1
