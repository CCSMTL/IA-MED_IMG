import os

import torch
import tqdm
import numpy as np
from custom_utils import dummy_context_mgr


def training_loop(
    model, loader, optimizer, criterion, device, minibatch_accumulate, scaler
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
    # Profiler =  torch.profiler.profile(
    #         # schedule=torch.profiler.schedule(
    #         #     wait=2,
    #         #     warmup=2,
    #         #     active=6,
    #         #     repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler("log"),
    #         with_stack=True
    # )
    Profiler=dummy_context_mgr()
    with Profiler as profiler:
        for inputs, labels in loader:
            # get the inputs; data is a list of [inputs, labels]

            inputs, labels = (
                inputs.to(device, non_blocking=True),
                labels.to(device, non_blocking=True),
            )

                # forward + backward + optimize
            with torch.cuda.amp.autocast():

                outputs = model(inputs)
                if model._get_name() == "Unet": #TODO : Find way to remove ugly if. Plus get_name doesnt work with DataParallel
                    inputs=inputs[:,0,:,:]
                    outputs=outputs[:,0,:,:]
                    labels = inputs

                assert not torch.isnan(outputs).any(), "Your outputs contain Nans!!!!"

                loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                running_loss += loss.detach()

                # gradient accumulation
                if i % minibatch_accumulate == 0:
                    i = 1

                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)

                    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5) #TODO : add norm c as hyperparameters
                    #optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                # ending loop
                del (
                    inputs,
                    labels,
                    loss,
                    outputs,
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

        # forward + backward + optimize
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            if model._get_name() == "Unet":
                inputs = inputs[:, 0, :, :]
                outputs = outputs[:, 0, :, :]
                labels = inputs
            loss = criterion(outputs, labels)
            running_loss += loss.detach()

        if inputs.shape != labels.shape:
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
    batch_size=1,
):

    epoch = 0
    train_loss_list = []
    val_loss_list = []
    best_loss = np.inf

    patience_init = patience
    pbar = tqdm.tqdm(total=epoch_max)

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    while patience > 0 and epoch < epoch_max:  # loop over the dataset multiple times

        train_loss = training_loop(
            model,
            tqdm.tqdm(training_loader, leave=False),
            optimizer,
            criterion,
            device,
            minibatch_accumulate,
            scaler,
        )
        val_loss, results = validation_loop(
            model, tqdm.tqdm(validation_loader, leave=False), criterion, device
        )

        # LOGGING DATA
        train_loss_list.append(train_loss / len(training_loader.dataset))
        val_loss_list.append(val_loss / len(validation_loader.dataset))
        if experiment:
            experiment.log_metric("training_loss", train_loss.tolist(), epoch=epoch)
            experiment.log_metric("validation_loss", val_loss.tolist(), epoch=epoch)

            for key in metrics:
                pred = results[1].numpy()
                true = results[0].numpy()
                experiment.log_metric(key, metrics[key](true, pred), epoch=epoch)

        if val_loss < best_loss:
            best_loss = val_loss

            experiment.save_weights(model)
            patience = patience_init
        else:
            patience -= 1
            print("patience has been reduced by 1")
        # Finishing the loop
        epoch += 1
        pbar.update(1)
    print("Finished Training")



