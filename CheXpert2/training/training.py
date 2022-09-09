import numpy as np
import torch
import torch.distributed as dist
import tqdm

from CheXpert2.custom_utils import set_parameter_requires_grad


def training_loop(
        model, loader, optimizer, criterion, device, scaler, clip_norm, autocast
):
    """

    :param model: model to train
    :param loader: training dataloader
    :param optimizer: optimizer
    :param criterion: criterion for the loss
    :param device: device to do the computations on
    :param minibatch_accumulate: number of minibatch to accumulate before applying gradient. Can be useful on smaller gpu memory
    :return: epoch loss, tensor of concatenated labels and predictions
    """
    running_loss = 0

    model.train()
    i = 0
    n = len(loader)
    for inputs, labels in loader:

        #send to GPU
        inputs, labels = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )

        #Apply transformation on GPU to avoid CPU bottleneck
        #inputs = loader.iterable.dataset.transform(inputs)
        inputs, labels = loader.iterable.dataset.advanced_transform((inputs, labels))
        inputs = loader.iterable.dataset.preprocess(inputs)

        with torch.cuda.amp.autocast(enabled=autocast):
            outputs = model(inputs)
            loss = criterion(outputs, labels)


        #assert not torch.isnan(outputs).any()
        # outputs = torch.nan_to_num(outputs,0)

        if autocast:
            scaler.scale(loss).backward()
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), clip_norm
        )
        if autocast:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        # optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.detach()
        # ending loop
        del (
            outputs,
            labels,
            inputs,
            loss,
        )  # garbage management sometimes fails with cuda
        i += 1


    return running_loss


@torch.no_grad()
def validation_loop(model, loader, criterion, device):
    """

    :param model: model to evaluate
    :param loader: dataset loader
    :param criterion: criterion to evaluate the loss
    :param device: device to do the computation on
    :return: val_loss for the N epoch, tensor of concatenated labels and predictions
    """
    running_loss = 0


    model.eval()

    results = [torch.tensor([]), torch.tensor([])]

    for inputs, labels in loader:
        # get the inputs; data is a list of [inputs, labels]

        inputs, labels = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        inputs = loader.iterable.dataset.preprocess(inputs)
        # forward + backward + optimize

        outputs = model(inputs)
        loss = criterion(outputs.float(), labels.float())

        running_loss += loss.detach()
        outputs = outputs.detach().cpu()
        results[1] = torch.cat((results[1], outputs), dim=0)
        results[0] = torch.cat((results[0], labels.cpu().round(decimals=0)),
                               dim=0)  # round to 0 or 1 in case of label smoothing

        del (
            inputs,
            labels,
            outputs,
            loss,
        )  # garbage management sometimes fails with cuda

    return running_loss, results,


def training(
    model,
    optimizer,
    criterion,
    training_loader,
    validation_loader,
    device="cpu",
    metrics=None,
    minibatch_accumulate=1,
    experiment=None,
    pos_weight = 1,
    clip_norm = 100,
    autocast=True
):
    epoch = 0
    results = None
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()
    val_loss = np.inf
    n, m = len(training_loader), len(validation_loader)
    criterion_val = criterion()
    criterion = criterion(pos_weight=torch.ones((len(experiment.names),),device=device)*pos_weight)

    position = device + 1 if type(device) == int else 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=2,T_mult=5)
    #scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1)
    while experiment.keep_training:  # loop over the dataset multiple times
        metrics_results = {}
        if dist.is_initialized():
            training_loader.sampler.set_epoch(epoch)

        train_loss = training_loop(
            model,
            tqdm.tqdm(training_loader, leave=False, position=position),

            optimizer,
            criterion,
            device,
            scaler,
            clip_norm,
            autocast
        )
        if experiment.rank == 0:
            val_loss, results = validation_loop(
                model, tqdm.tqdm(validation_loader, position=position, leave=False), criterion_val, device
            )
            val_loss = val_loss.cpu() / m
            if metrics:
                for key in metrics:
                    pred = results[1].numpy()
                    true = results[0].numpy().round(0)
                    metric_result = metrics[key](true, pred)
                    metrics_results[key] = metric_result

                experiment.log_metrics(metrics_results, epoch=epoch)
                experiment.log_metric("training_loss", train_loss.cpu() / n, epoch=epoch)
                experiment.log_metric("validation_loss", val_loss, epoch=epoch)

            # Finishing the loop

        experiment.next_epoch(val_loss, model)
        scheduler.step()
        # if not dist.is_initialized() and experiment.epoch % 5 == 0:
        #     set_parameter_requires_grad(model, 1 + experiment.epoch // 2)
        if experiment.epoch == experiment.epoch_max:
            experiment.keep_training = False

    print("Finished Training")

    return results
