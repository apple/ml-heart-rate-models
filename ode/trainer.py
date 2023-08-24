#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import collections
import time
from typing import Optional

import pandas as pd
import torch
from torch import optim
import tqdm
import numpy as np
from torch.utils.data import DataLoader

from ode.ode import ODEModel

STD_HR = 5  # wearables heart rate measurements usually have a standard deviation around ±5 bpm
STD_EMBEDDING = 1.0  # we regularize the embeddings to have a standard deviation of 1.0


def l2_reg(params):
    """Compute the L2 regularization term for a list of parameters."""
    return sum([(p**2).sum() for p in params])


def l2_error(tensor1, tensor2=0.0, std=1.0):
    """Compute the L2 error between two tensors, with optional standardization."""
    return ((tensor1 - tensor2) / std).pow(2).sum()


def train_ode_model(
    model: ODEModel,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    train_workout_ids: Optional[set] = None,
):
    """Train the ODE model.

    Parameters
    ----------
    model : ODEModel
        The model to train.
    train_dataloader : DataLoader
        The dataloader for the training data.
    test_dataloader : DataLoader
        The dataloader for the test data (can also contain training data if train_workout_ids is given).
    train_workout_ids : set, optional
        The set of workout ids that are used for training. Used to compute the train/test loss.
    """
    ode_config = model.config
    optimizer = optim.Adam(model.parameters(), lr=ode_config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=4)
    evaluation_logs = []

    for epoch in range(ode_config.n_epochs):
        start = time.time()
        epoch_loss = 0
        model.train()

        for batch in tqdm.tqdm(train_dataloader):
            predictions = model.forecast_batch(
                activity=batch["activity"],
                times=batch["time"],
                workout_id=batch["workout_id"],
                subject_id=batch["subject_id"],
                history=batch["history"],
                history_length=batch["history_length"],
                weather=batch["weather"],
                step_size=ode_config.ode_step_size,
            )
            predictions_hr = predictions["heart_rate"]
            heart_rate_reconstruction_l2 = l2_error(
                predictions_hr, batch["heart_rate"], std=STD_HR
            )

            embedding_l2 = (
                (predictions["workout_embedding"] / STD_EMBEDDING).pow(2).sum()
            )
            embedding_l2 *= ode_config.embedding_reg_strength
            decoders_weights_l2 = l2_reg(model.fatigue_fn.parameters())
            if model.weather_fn is not None:
                decoders_weights_l2 += l2_reg(model.weather_fn.parameters())
            decoders_weights_l2 += l2_reg(model.activity_fn.parameters())
            decoders_weights_l2 *= ode_config.decoder_reg_strength

            if model.embedding_store.encoder is not None:
                encoder_weights_l2 = l2_reg(model.embedding_store.encoder.parameters())
                encoder_weights_l2 *= ode_config.encoder_reg_strength
            else:
                encoder_weights_l2 = 0
            loss = (
                heart_rate_reconstruction_l2
                + embedding_l2
                + decoders_weights_l2
                + encoder_weights_l2
            )
            optimizer.zero_grad()
            loss.backward()
            if ode_config.clip_gradient > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ode_config.clip_gradient
                )

            # check for nans in gradients
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any():
                    print(f"NaN in gradient for {name}")
                    print(param.grad)
                    raise ValueError("NaN in gradient")
            optimizer.step()
            epoch_loss += loss.item()

        # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        # torch.save(checkpoint, path + f"pickle/epoch{epoch}_checkpoint.pth")

        evaluation_log = evaluate(
            model, epoch, test_dataloader, start, train_workout_ids
        )
        evaluation_logs.append(evaluation_log)

        training_loss = evaluation_log[evaluation_log["in_train"]]["l1"].mean()
        scheduler.step(training_loss)

    return evaluation_logs


def evaluate(model: ODEModel, epoch, test_dataloader, start_time, train_workout_ids):
    ode_config = model.config
    model.eval()
    with torch.no_grad():
        predicted_hr_all = []
        true_hr_all = []
        embeddings_all = []
        ode_parameters_all = collections.defaultdict(list)
        subject_id_all = []
        workout_id_all = []
        for batch in tqdm.tqdm(test_dataloader):
            predictions = model.forecast_batch(
                batch["activity"],
                batch["time"],
                batch["workout_id"],
                batch["subject_id"],
                batch["history"],
                batch["history_length"],
                batch["weather"],
                step_size=ode_config.ode_step_size,
            )
            predictions_hr = predictions["heart_rate"].detach().numpy()
            for ii in range(len(batch["full_workout_length"])):
                end_index = int(batch["full_workout_length"][ii].item())
                predicted_hr_all.append(predictions_hr[ii, :end_index])
                true_hr_all.append(batch["heart_rate"][ii, :end_index].detach().numpy())
            embeddings_all.append(predictions["workout_embedding"])
            for k in predictions["ode_params"]:
                ode_parameters_all[k].append(predictions["ode_params"][k])
            subject_id_all.extend(batch["subject_id"])
            workout_id_all.extend(batch["workout_id"])
    embeddings_all = torch.cat(embeddings_all, dim=0)
    for k in ode_parameters_all:
        # noinspection PyTypeChecker
        ode_parameters_all[k] = torch.cat(ode_parameters_all[k], dim=0)

    predictions_to_save = dict()
    metrics = {
        "l2": lambda x, y: ((x - y) ** 2).mean().item() ** 0.5,
        "l1": lambda x, y: (np.abs(x - y)).mean().item(),
        "relative": lambda pred, truth: (np.abs(pred - truth) / truth).mean().item(),
    }

    logged_data_for_all_workouts = []
    for idx in tqdm.tqdm(range(len(predicted_hr_all))):
        logged_data_for_workout = {}
        for m in metrics:
            logged_data_for_workout[m] = metrics[m](
                predicted_hr_all[idx], true_hr_all[idx]
            )
            logged_data_for_workout[m + "-after2min"] = metrics[m](
                predicted_hr_all[idx][12:], true_hr_all[idx][12:]
            )

        logged_data_for_workout["in_train"] = workout_id_all[idx] in train_workout_ids
        logged_data_for_workout["subject_id"] = subject_id_all[idx]
        logged_data_for_workout["workout_id"] = workout_id_all[idx]
        logged_data_for_workout[f"subject_embeddings"] = (
            embeddings_all[idx, : model.config.subject_embedding_dim].detach().numpy()
        )
        logged_data_for_workout[f"encoder_embeddings"] = (
            embeddings_all[idx, model.config.subject_embedding_dim :].detach().numpy()
        )
        for ode_param_name in ode_parameters_all:
            logged_data_for_workout[ode_param_name] = ode_parameters_all[
                ode_param_name
            ][idx].item()
        predictions_to_save[workout_id_all[idx]] = np.array(
            [true_hr_all[idx], predicted_hr_all[idx]]
        )
        logged_data_for_all_workouts.append(logged_data_for_workout)

    logged_data_for_all_workouts = pd.DataFrame(logged_data_for_all_workouts)
    train_flag = logged_data_for_all_workouts["in_train"]

    print(
        f"Epoch {epoch} took {time.time() - start_time:.1f} seconds",
        "Train mean l1: %.3f bpm (= %.3f %%)"
        % (
            logged_data_for_all_workouts[train_flag]["l1"].mean(),
            logged_data_for_all_workouts[train_flag]["relative"].mean() * 100,
        ),
        "Test mean l1: %.3f bpm (= %.3f %%)"
        % (
            logged_data_for_all_workouts[~train_flag]["l1"].mean(),
            logged_data_for_all_workouts[~train_flag]["relative"].mean() * 100,
        ),
        "Test mean l1-after2min: %.3f bpm (= %.3f %%)"
        % (
            logged_data_for_all_workouts[~train_flag]["l1-after2min"].mean(),
            logged_data_for_all_workouts[~train_flag]["relative-after2min"].mean()
            * 100,
        ),
        sep="\n",
    )
    return logged_data_for_all_workouts
