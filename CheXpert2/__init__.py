
#nc: 15

#names = [ "Cardiomegaly", "Edema","Consolidation", "Atelectasis", "Pleural Effusion","No Finding"]

names= ["Opacity","Air","Liquid","Cardiomegaly","Lung Lesion" ,"Emphysema","Edema"    ,"Consolidation"  ,"Atelectasis"    ,"Pneumothorax"    ,"Pleural Effusion"    ,"Fracture" ,"Hernia","Infiltration","Mass","Nodule","Pneumo other","No Finding"]

debug_config = {
        "model": "convnext_tiny",
        "batch_size": 2,
        "img_size": 224,
        "num_worker": 0,
        "augment_prob" : [0,]*6,
        "augment_intensity": 0,
        "cache": False,
        "N": 0,
        "M": 2,
        "clip_norm": 100,
        "label_smoothing": 0,
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "freeze": False,
        "pretrained": False,
        "pretraining": 0,
        "channels": 1,
        "autocast": True,
        "use_frontal" : False,
        "pos_weight": 1,
    }