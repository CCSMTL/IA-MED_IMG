import numpy as np
import torch
import torch.distributed as dist
import tqdm


def training_loop(
    model, loader, optimizer, criterion, device, minibatch_accumulate, scaler,clip_norm
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
    i = 1

    for inputs, labels in loader:

        # get the inputs; data is a list of [inputs, labels]

        # forward + backward + optimize
        # loss = training_core(model, inputs, scaler, criterion,device)

        inputs, labels = (
            inputs.to(device, non_blocking=True, memory_format=torch.channels_last),
            labels.to(device, non_blocking=True),
        )
        inputs = loader.iterable.dataset.transform(inputs)
        inputs, labels = loader.iterable.dataset.advanced_transform((inputs, labels))
        inputs = loader.iterable.dataset.preprocess(inputs)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)

            # assert not torch.isnan(outputs).any(), "Your outputs contain Nans!!!!"

            loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
        running_loss += loss.detach()

        # gradient accumulation
        if i % minibatch_accumulate == 0:
            i = 1

            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip_norm
            )
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
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
            inputs.to(device, non_blocking=True, memory_format=torch.channels_last),
            labels.to(device, non_blocking=True),
        )
        inputs = loader.iterable.dataset.preprocess(inputs)
        # forward + backward + optimize
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += torch.mean(loss.detach().cpu())

        if inputs.shape != labels.shape:  # prevent storing images if training unets
            results[1] = torch.cat(
                (results[1], torch.sigmoid(outputs).detach().cpu()), dim=0
            )
            results[0] = torch.cat((results[0], labels.cpu()), dim=0)

        del (
            inputs,
            labels,
            outputs,
            loss,
        )  # garbage management sometimes fails with cuda

    return running_loss, results


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
    patience=5,
    epoch_max=50,
    clip_norm = 100
):
    epoch = 0
    train_loss_list = []
    val_loss_list = []
    best_loss = np.inf

    patience_init = patience

    pbar = tqdm.tqdm(total=epoch_max, position=device * 2)

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()
    n, m = len(training_loader.dataset), len(validation_loader.dataset)
    while patience > 0 and epoch < epoch_max:  # loop over the dataset multiple times
        metrics_results = {}
        if dist.is_initialized():
            training_loader.sampler.set_epoch(epoch)
        train_loss = training_loop(
            model,
            tqdm.tqdm(training_loader, leave=False, position=device * 3),
            optimizer,
            criterion,
            device,
            minibatch_accumulate,
            scaler,
            clip_norm
        )
        if experiment.rank == 0:
            val_loss, results = validation_loop(
                model, tqdm.tqdm(validation_loader, leave=False), criterion, device
            )
            # LOGGING DATA
            train_loss_list.append(train_loss / n)
            val_loss_list.append(val_loss / m)

            if metrics:
                for key in metrics:
                    pred = results[1].numpy()
                    true = results[0].numpy()
                    metric_result = metrics[key](true, pred)
                    metrics_results[key] = metric_result
                    experiment.log_metric(key, metric_result, epoch=epoch)

            experiment.log_metric("training_loss", train_loss.tolist(), epoch=epoch)
            experiment.log_metric("validation_loss", val_loss.tolist(), epoch=epoch)

            if val_loss < best_loss or epoch == 0:

                best_loss = val_loss

                experiment.save_weights(model)
                patience = patience_init
                if metrics:
                    summary = metrics_results

                else:
                    summary = {}
            else:
                patience -= 1
                print("patience has been reduced by 1")
        # Finishing the loop
        if dist.is_initialized():
            if dist.get_rank() == 0:
                pbar.update(1)
                for i in range(torch.cuda.device_count() - 1):
                    dist.isend(torch.tensor([patience]), i + 1)
            else:
                patience = torch.tensor([0])
                dist.irecv(patience, src=0)
                patience = patience.item()
        else:
            pbar.update(1)
        epoch += 1

    print("Finished Training")

    return results,summary
