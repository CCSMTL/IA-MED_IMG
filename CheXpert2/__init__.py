
#nc: 15

#names = [ "Cardiomegaly", "Edema","Consolidation", "Atelectasis", "Pleural Effusion","No Finding"]

names= ["Opacity","Air","Liquid","Cardiomegaly","Lung Lesion" ,"Emphysema","Edema"    ,"Consolidation"  ,"Atelectasis"    ,"Pneumothorax"    ,"Pleural Effusion"    ,"Fracture" ,"Hernia","Infiltration","Mass","Nodule","Pleural Other","No Finding"]

debug_config = {
        "model": "convnext_tiny",
        "batch_size": 32,
        "img_size": 224,
        "num_worker": 0,
        "augment_prob" : [0,1,0.5,1,0,0],
        "augment_intensity": 0,
        "cache": False,
        "clip_norm": 1,
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
        "debug" : True
    }