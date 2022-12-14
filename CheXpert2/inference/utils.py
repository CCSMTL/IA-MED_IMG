
import torch
from CheXpert2.models.CNN import CNN
import warnings
import tqdm


@torch.no_grad()
def infer_loop(model, loader, criterion, device):
    """

    :param model: model to evaluate
    :param loader: dataset loader
    :param criterion: criterion to evaluate the loss
    :param device: device to do the computation on
    :return: val_loss for the N epoch, tensor of concatenated labels and predictions
    """
    running_loss = 0
    results = [torch.tensor([]), torch.tensor([])]

    for inputs, labels,idx in tqdm.tqdm(loader):
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        #inputs,labels = loader.dataset.advanced_transform((inputs, labels))
        # forward + backward + optimize

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        outputs = torch.sigmoid(outputs)
        running_loss += loss.detach()

        if inputs.shape != labels.shape:  # prevent storing images if training unets
            results[1] = torch.cat(
                (results[1], outputs.detach().cpu()), dim=0
            )
            results[0] = torch.cat((results[0], labels.cpu()), dim=0)

        del (
            inputs,
            labels,
            outputs,
            loss,
        )  # garbage management sometimes fails with cuda


    return running_loss, results



def load_model(model_name,weight_path) :
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
        warnings.warn("No gpu is available for the computation")
    models = [
        CNN(model_name, img_size=384, channels=3, num_classes=18, pretrained=False,
            pretraining=False),
        #    CNN("convnext_base", img_size=384, channels=1, num_classes=14, pretrained=False, pretraining=False),
        #    CNN("densenet201", img_size=384, channels=1, num_classes=14, pretrained=False, pretraining=False),
        #    CNN("densenet201", img_size=384, channels=1, num_classes=14, pretrained=False, pretraining=False),
    ]
    # model =  torch.nn.parallel.DistributedDataParallel(model)

    # api = wandb.Api()
    # run = api.run(f"ccsmtl2/Chestxray/{args.run_id}")
    # run.file("models_weights/convnext_base/DistributedDataParallel.pt").download(replace=True)
    weights = [
        weight_path,
        #    "/data/home/jonathan/IA-MED_IMG/models_weights/convnext_base_2.pt",
        #    "/data/home/jonathan/IA-MED_IMG/models_weights/densenet201.pt",
        #    "/data/home/jonathan/IA-MED_IMG/models_weights/densenet201_2.pt",
    ]

    for model, weight in zip(models, weights):
        state_dict = torch.load(weight, map_location=torch.device(device))

        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
        #     new_state_dict[name] = v

        # model.load_state_dict(new_state_dict)
        model.load_state_dict(state_dict)
        # model = model.to(device)
        model.eval()
        model = model.to(device)
        return model