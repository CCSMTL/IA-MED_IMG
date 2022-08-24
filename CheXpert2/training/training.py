import numpy as np
import torch
import torch.distributed as dist
import tqdm


def training_loop(
        model, loader, optimizer, criterion, device, scaler, clip_norm,autocast
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

        # get the inputs; data is a list of [inputs, labels]

        # forward + backward + optimize
        # loss = training_core(model, inputs, scaler, criterion,device)

        inputs, labels = (
            inputs.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        inputs = loader.iterable.dataset.transform(inputs)
        inputs, labels = loader.iterable.dataset.advanced_transform((inputs, labels))
        inputs = loader.iterable.dataset.preprocess(inputs)

        with torch.cuda.amp.autocast(enabled=autocast):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
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
        inputs = loader.dataset.preprocess(inputs)
        # forward + backward + optimize

        outputs = model(inputs)
        loss = criterion(outputs.float(), labels.float())

        running_loss += loss.detach()
        outputs = outputs.detach().cpu()
        outputs[:, [0, 1, 2, 6, 8, 9, 10, 11, 12, 13]] = torch.sigmoid(
            outputs[:, [0, 1, 2, 6, 8, 9, 10, 11, 12, 13]])  # .clone()
        outputs[:, [3, 4, 5, 7]] = torch.softmax(outputs[:, [3, 4, 5, 7]], dim=1).clone()
        # outputs = torch.sigmoid(outputs).detach().cpu()
        # outputs[:, [8,9,10]] = torch.softmax(outputs[:, [8,9,10]], dim=1).clone()
        # outputs[:,[0,2,8,9,10,11,12]] = torch.mul(outputs[:,[0,2,8,9,10,11,12]].clone(),outputs[:,13].clone()[:,None])
        outputs[:, 1] = torch.mul(outputs[:, 1], outputs[:, 0])
        outputs[:, [3, 4, 5, 7]] = torch.mul(outputs[:, [3, 4, 5, 7]], outputs[:, 2][:, None])
        outputs[:, 6] = torch.mul(outputs[:, 5], outputs[:, 6])

        # outputs[:, 13] = 1 - outputs[:, 13]  # lets the model predict sick instead of no finding

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

    clip_norm = 100,
    autocast=True
):
    epoch = 0
    results = None
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()
    val_loss = np.inf
    n, m = len(training_loader), len(validation_loader)

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
                model, validation_loader, criterion, device
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
        if experiment.epoch == experiment.epoch_max:
            experiment.keep_training = False

    print("Finished Training")

    return results
