
#nc: 15

names = [ "Cardiomegaly", "Edema","Consolidation", "Atelectasis", "Pleural Effusion","No Finding"]

#names= ["Opacity","Air","Liquid","Cardiomegaly","Lung Lesion" ,"Emphysema","Edema","Consolidation"  ,"Atelectasis"    ,"Pneumothorax"    ,"Pleural Effusion"    ,"Fracture" ,"Hernia","Infiltration","Mass","Nodule","Pleural Other","No Finding"]

debug_config = {
        "model": "densenet121",
        "batch_size": 2,
        "img_size": 224,
        "num_worker": 0,
        "augment_prob" : [1,1,1,1,1],
        "cache": False,
        "clip_norm": 1,
        "label_smoothing": 0.05,
        "lr": 0.001,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0.01,
        "freeze": False,
        "pretrained": True,
        "pretraining": 0,
        "channels": 1,
        "autocast": False,
        "use_frontal" : False,
        "pos_weight": 1,
        "debug" : True,
        "global_pool" : "avg",
    }