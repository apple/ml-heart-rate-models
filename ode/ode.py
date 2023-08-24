#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import datetime
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn

from ode.modules_cnn import CNNEncoder
from ode.data import WorkoutDatasetConfig
from torchdiffeq import odeint
from ode.modules_dense_nn import PersonalizedScalarNN, DenseNN, PersonalizationType

EPSILON = 1e-3


@dataclass
class OdeConfig:
    # noinspection PyUnresolvedReferences
    """
    Configuration of a OdeModel.

    Attributes
    ----------
    data_config: WorkoutDatasetConfig
        Configuration of the dataset.
    learning_rate: float, default=1e-3
        Learning rate of the optimizer.
    n_epochs: int, default=50
        Number of epochs to train.
    seed: int, default=0
        Random seed.
    ode_step_size: float, default=1.0
        Step size of the ODE solver. 0.5 is a bit more stable than 1 but slower.
    clip_gradient: float, default=5.0
        Clip the gradient norm to this value. 0 to disable.
    subject_embedding_dim: int, default=8
        Dimension of the subject specific embedding.
    encoder_embedding_dim: int, default=8
        Dimension of the embedding of the workout history encoder.
    encoder_kernel_size: int, default=6
        Kernel size of the causal CNN encoder.
    encoder_layers: str, default="128"
        Number of channels of the causal CNN encoder. The string is of the form "dim_layer1,dim_layer_2,dim_layer3".
    embedding_reg_strength: float, default=1.0
        L2 regularization strength of the embeddings.
    decoder_reg_strength: float, default=0.0
        L2 regularization strength of the decoder networks: `activity_fn`, `weather_fn`, `fatigue_fn`.
    encoder_reg_strength: float, default=0.0
        L2 regularization strength of the encoder.
    ode_parameter_layers: str, default="32,8"
        Number of hidden units of the networks which decode the embeddings into the ODE parameters.
    ode_parameter_activation: str, default="softplus"
        Activation function of the networks which decode the embeddings into the ODE parameters.
    activity_fn_layers: str, default="128,64"
        Number of hidden units of the network which decodes the ODE parameter `I` into the demand `f(I)`.
    activity_fn_activation: str, default="softplus"
        Activation function of the network which decodes the ODE parameter `I` into the demand `f(I)`.
    activity_fn_embedding_personalization: ["none", "softmax", "concatenate"], default="softmax"
        Personalization of the demand function with the embedding.
        - "none": no personalization.
        - "softmax": there are `embedding_dim` different demand functions which are averaged with weights given by a
          softmax of the embedding.
        - "concatenate": the embedding is concatenated to the input of the demand function.
    weather_fn_embedding_personalization: ["none", "softmax", "concatenate"], default="none"
        Personalization of the weather function with the embedding. See `activity_fn_embedding_personalization`.
    fatigue_fn_embedding_personalization: ["none", "softmax", "concatenate"], default="none"
        Personalization of the fatigue function with the embedding. See `activity_fn_embedding_personalization`.
    ranges_A_B_alpha_beta: str, default="-3 5, -3 5, 0.1 3, 0.1 3"
        Ranges of the ODE parameters `A`, `B`, `alpha`, `beta`.
        The string is of the form "A_min A_max, B_min B_max, alpha_min alpha_max, beta_min beta_max".
    range_activity_fn: str, default="30 250"
        Range of the demand function. The string is of the form "min max".
    python_start: str
        Date and time automatically generated when the model is initialized.
    """

    data_config: WorkoutDatasetConfig

    # training
    learning_rate: float = 1e-3
    n_epochs: int = 50
    seed: int = 0
    ode_step_size: float = 1.0
    clip_gradient: float = 5.0

    # embeddings
    subject_embedding_dim: int = 8
    encoder_embedding_dim: int = 8
    encoder_kernel_size: int = 6
    encoder_layers: str = "128"

    # regularization
    embedding_reg_strength: float = 1.0
    decoder_reg_strength: float = 0.0
    encoder_reg_strength: float = 0.0

    # architecture of the networks
    ode_parameter_layers: str = "32,8"
    ode_parameter_activation: str = "softplus"
    activity_fn_layers: str = "128,64"
    activity_fn_activation: str = "softplus"

    # personalization with embeddings
    activity_fn_embedding_personalization: PersonalizationType = "softmax"
    weather_fn_embedding_personalization: PersonalizationType = "none"
    fatigue_fn_embedding_personalization: PersonalizationType = "none"

    # ranges of the parameters
    ranges_A_B_alpha_beta: str = "-3 5, -3 5, 0.1 3, 0.1 3"
    range_activity_fn: str = "30 250"

    python_start: str = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")


class EmbeddingStore(nn.Module):
    """
    Manage the embeddings for the ODE models.

    Parameters
    ----------
    ode_config : OdeConfig
    workouts_info: pd.DataFrame
        The data in DataFrame format containing the subject_id and workout_id columns in order to prepare the correct
        number of embeddings and later map each workout to the correct subject embedding.

    Methods
    -------
    initialize_subject_embeddings()
        initialize the subject specific embeddings.
    initialize_encoder()
        initialize the encoder.
    get_embeddings_from_workout_ids(workout_ids, auto_encoder_input=None, auto_encoder_length=None)
        compute the embedding from the list of workout ids. requires the auto_encoder input (and lengths)
        to compute the auto-encoder embeddings.
    """

    def __init__(
        self,
        ode_config: OdeConfig,
        workouts_info: pd.DataFrame,
    ):
        super().__init__()
        # subject embedding parameters
        self.subject_id_column = ode_config.data_config.subject_id_column
        self.workout_id_column = ode_config.data_config.workout_id_column
        self.workouts_info = workouts_info[
            [self.subject_id_column, self.workout_id_column]
        ]
        self.subject_embedding_dim = ode_config.subject_embedding_dim
        self.subject_id_to_embedding_index = None
        self.workout_id_to_embedding_index = None
        self.n_subject_embeddings = None
        self.subject_embeddings = None
        self.initialize_subject_embeddings()

        # encoder embedding parameters
        self.encoder_input_dim = ode_config.data_config.history_dim()
        self.encoder_embedding_dim = ode_config.encoder_embedding_dim
        self.encoder_kernel_size = ode_config.encoder_kernel_size
        self.encoder_layers = list(map(int, ode_config.encoder_layers.split(",")))
        self.encoder = None
        self.initialize_encoder()

        self.dim_embedding = self.subject_embedding_dim + self.encoder_embedding_dim

    def initialize_subject_embeddings(self):
        """
        Initialize one embedding per subject id and prepare the mapping from workout id to embedding index.
        """
        if self.subject_embedding_dim is None or self.subject_embedding_dim == 0:
            return
        unique_subject_ids = self.workouts_info[self.subject_id_column].unique()
        self.n_subject_embeddings = len(unique_subject_ids)
        self.subject_id_to_embedding_index = {
            s_id: idx for idx, s_id in enumerate(unique_subject_ids)
        }
        self.workout_id_to_embedding_index = dict()
        for s_id, w_id in self.workouts_info[
            [self.subject_id_column, self.workout_id_column]
        ].values:
            self.workout_id_to_embedding_index[
                w_id
            ] = self.subject_id_to_embedding_index[s_id]
        self.subject_embeddings = nn.Embedding(
            self.n_subject_embeddings, self.subject_embedding_dim, max_norm=5.0
        )

    def initialize_encoder(self):
        """
        Initialize the encoder.
        The encoder is a CNN which takes as input the history of workouts and outputs an embedding.
        """
        if self.encoder_embedding_dim is None or self.encoder_embedding_dim <= 0:
            return
        dims = [self.encoder_input_dim, *self.encoder_layers]
        self.encoder = CNNEncoder(
            *dims,
            dim_output_embedding=self.encoder_embedding_dim,
            kernel_size=self.encoder_kernel_size,
        )

    def get_embeddings_from_workout_ids(
        self, workout_ids, history=None, history_lengths=None
    ):
        embeddings = []
        if self.subject_embeddings is not None:
            subject_embeddings = self.subject_embeddings(
                torch.LongTensor(
                    [self.workout_id_to_embedding_index[wid] for wid in workout_ids]
                )
            )
            embeddings.append(subject_embeddings)
        if self.encoder is not None:
            encoded_embeddings = self.encoder(history, history_lengths)
            embeddings.append(encoded_embeddings)

        embeddings = torch.cat(embeddings, dim=-1)
        return embeddings


def get_activation(name):
    return {"softplus": nn.Softplus(), "tanh": nn.Tanh(), "relu": nn.ReLU()}[name]


class ODEModel(nn.Module):
    """
    The ODE model.
    The forward method is compatible with `torchdiffeq.odeint` method for differentiable integration.

    Parameters
    ----------
    workouts_info: pd.DataFrame
        The data in DataFrame format containing the subject_id and workout_id columns in order to prepare
        the embeddings. Given to the EmbeddingStore class.
    config: OdeConfig
        Configuration of the model.

    Methods
    -------
    forward(t, x)
        time t is a scalar and x is of shape (batch_size, 2). x[:,0] is HR, x[:,1] is D
        forward returns the derivative for [HR,D] at time t for point x.

    initialize_batch(
        activity,
        times,
        workout_id,
        subject_id,
        history=None,
        history_length=None,
        weather=None,
    ):
        Pre-compute the embeddings, the intensity function f(I) during each workout, the weather and the fatigue,
        so that solving the ODE is faster.

    forecast_batch(
        activity,
        times,
        workout_id,
        subject_id,
        history=None,
        history_length=None,
        weather=None,
        step_size=1.0,
    )
        Forecast the HR for the batch. This is the function that one should use to make prediction (for train or test).
        It calls initialize_batch and then solves the ODE using torchdiffeq.odeint.
    """

    def __init__(
        self,
        workouts_info,
        config: OdeConfig,
    ):
        super().__init__()
        self.config = config
        torch.manual_seed(self.config.seed)

        self.dim_activity = self.config.data_config.n_activity_channels()
        self.ode_parameter_functions = nn.ModuleDict()
        self.embedding_store = EmbeddingStore(
            self.config,
            workouts_info,
        )
        self.dim_embedding = self.embedding_store.dim_embedding
        self.ode_parameter_layers = [
            int(d) for d in self.config.ode_parameter_layers.split(",")
        ]

        # noinspection PyTypeChecker
        parameter_ranges = list(
            map(
                float,
                filter(
                    len, self.config.ranges_A_B_alpha_beta.replace(",", " ").split(" ")
                ),
            )
        )

        for parameter_name, low, high in [
            ("A", *parameter_ranges[0:2]),
            ("B", *parameter_ranges[2:4]),
            ("alpha", *parameter_ranges[4:6]),
            ("beta", *parameter_ranges[6:8]),
            ("hr_min", 40.0, 90.0),
            ("hr_max", 140.0, 210.0),
        ]:
            self.ode_parameter_functions[parameter_name] = PersonalizedScalarNN(
                self.dim_embedding,
                *self.ode_parameter_layers,
                personalization="none",
                output_bounds=(low, high),
                activation=get_activation(self.config.ode_parameter_activation),
            )

        fatigue_layers = [1, 32, 16]
        self.fatigue_fn = PersonalizedScalarNN(
            *fatigue_layers,
            personalization=self.config.fatigue_fn_embedding_personalization,
            dim_personalization=self.dim_embedding,
            output_bounds=(0.5, 1.5),
        )

        weather_layers = [self.config.data_config.n_weather_channels(), 32, 16]
        # pytorch issue a warning if the first layer is 0 (no weather data). it would work fine though.
        if weather_layers[0] == 0:
            self.weather_fn = None
        else:
            self.weather_fn = PersonalizedScalarNN(
                *weather_layers,
                personalization=self.config.weather_fn_embedding_personalization,
                dim_personalization=self.dim_embedding,
                output_bounds=(0.5, 1.5),
            )

        activity_fn_layers = [self.dim_activity] + [
            int(d) for d in self.config.activity_fn_layers.split(",")
        ]
        min_activity_fn, max_activity_fn = map(
            float, self.config.range_activity_fn.split(" ")
        )
        self.activity_fn = PersonalizedScalarNN(
            *activity_fn_layers,
            personalization=self.config.activity_fn_embedding_personalization,
            dim_personalization=self.dim_embedding,
            output_bounds=(min_activity_fn, max_activity_fn),
            activation=get_activation(self.config.activity_fn_activation),
        )

        # use embedding to estimate initial heart rate and initial demand
        self.initial_heart_rate_activity_fn = DenseNN(
            *[self.dim_embedding, 32, 2], output_bounds=(50.0, 200.0)
        )

        self._activity = None
        self._initial_heart_rate = None
        self._time = None
        self._workout_id = None
        self._subject_id = None
        self._initial_demand = None
        self._activity_coefficient = None
        self._fatigue_coefficient = None
        self._weather_coefficient = None
        self._intensity = None
        self._workout_embeddings = None
        self._weather = None
        self._ode_params = None
        self._initial_heart_rate_and_demand = None

        self._number_of_forward_calls = 0

    def forward(self, t, x):
        """
        Compute the derivative of the ODE at time t and for input x.

        Parameters
        ----------
        t: FloatTensor (of shape (,))
            time in the ODE, used to compute f(I(t)). The time is scaled to be time N on measurement index N.
            So the activity levels at time t are: f(I)[int(t)]. We replace int(t) by t.long() in torch.
        x: FloatTensor (of shape (batch_size, 2))
            contains the current values of HR and D.

        Returns
        -------
        stacked_derivative: FloatTensor (same shape as x)
            ODE derivatives at x and time t for the batch
        """
        hr = x[..., 0]
        demand = x[..., 1]

        t = max(0, min(t.long(), self._intensity.shape[1] - 1))
        intensity = self._intensity[:, t]

        f_min = (
            (torch.abs(hr - self._ode_params["hr_min"] + EPSILON) + EPSILON) / 60.0
        ) ** self._ode_params["alpha"]
        f_max = (
            (torch.abs(self._ode_params["hr_max"] - hr + EPSILON) + EPSILON) / 60.0
        ) ** self._ode_params["beta"]

        # noinspection PyTypeChecker
        hr_dot = torch.where(
            hr >= self._ode_params["hr_max"],
            self._ode_params["hr_max"] - hr,
            torch.where(
                hr <= self._ode_params["hr_min"],
                self._ode_params["hr_min"] - hr,
                torch.exp(self._ode_params["A"]) * f_min * f_max * (demand - hr) / 60.0,
            ),
        )

        demand_dot = (intensity - demand) / 60.0 * torch.exp(self._ode_params["B"])

        # Alternative (simpler) models
        # hr_dot = torch.exp(self._ode_params["A"]) * (demand - hr) / 60.0
        # hr_dot = (intensity - hr) / 60.0 * torch.exp(self._ode_params["B"])
        # demand_dot = torch.zeros_like(hr_dot)

        stacked_derivative = torch.stack([hr_dot, demand_dot], dim=-1)
        self._number_of_forward_calls += 1
        return stacked_derivative

    def initialize_batch(
        self,
        activity,
        times,
        workout_id,
        subject_id,
        history=None,
        history_length=None,
        weather=None,
    ):
        """
        Where most of the computation happens to prepare the ODE solving.
        Compute the embeddings, the ODE parameters, pre-transform the activity measurements I into f(I), the weather
        function g(W) and the fatigue function h(t). Estimate the initial heart rates and demand.

        See `forecast_batch` for the description of the parameters.
        """
        workout_embeddings = self.embedding_store.get_embeddings_from_workout_ids(
            workout_id, history, history_length
        )
        ode_params = {
            k: self.ode_parameter_functions[k](workout_embeddings)
            for k in self.ode_parameter_functions
        }
        workout_embeddings_tiled = torch.tile(
            workout_embeddings.unsqueeze(1), (1, activity.shape[1], 1)
        )

        fatigue_coefficient = self.fatigue_fn(
            times.unsqueeze(-1), workout_embeddings_tiled
        )
        if self.weather_fn is not None:
            weather_coefficient = self.weather_fn(weather, workout_embeddings)
        else:
            weather_coefficient = 1.0

        activity_coefficient = self.activity_fn(
            activity, workout_embeddings_tiled
        ).squeeze(-1)

        intensity = activity_coefficient * fatigue_coefficient * weather_coefficient
        initial_heart_rate_and_demand = self.initial_heart_rate_activity_fn(
            workout_embeddings
        )

        # cache computations
        self._activity = activity
        self._workout_id = workout_id
        self._subject_id = subject_id
        self._time = times
        self._weather = weather
        self._workout_embeddings = workout_embeddings
        self._ode_params = ode_params
        self._fatigue_coefficient = fatigue_coefficient
        self._weather_coefficient = weather_coefficient
        self._activity_coefficient = activity_coefficient
        self._intensity = intensity
        self._initial_heart_rate_and_demand = initial_heart_rate_and_demand

    def forecast_batch(
        self,
        activity,
        times,
        workout_id,
        subject_id,
        history=None,
        history_length=None,
        weather=None,
        step_size=1.0,
    ):
        """
        Endpoint to solve the ODE from the data yielded by the WorkoutDataset.
        Will call initialize_batch and then use torchdiffeq.odeint to solve the ODE.

        Parameters
        ----------
        activity: FloatTensor
            Activity measurements I (batch_size, sequence_length, dim_activity)
        times: FloatTensor
            Time measurements t (batch_size, sequence_length)
        workout_id: LongTensor
            Workout ids (batch_size)
        subject_id: LongTensor
            Subject ids (batch_size)
        history: FloatTensor
            History of workouts (batch_size, history_length, dim_history)
        history_length: LongTensor
            Length of history (batch_size)
        weather: FloatTensor
            Weather measurements W (batch_size, sequence_length, dim_weather)
        step_size: float
            Step size for the ODE solver

        Returns
        ----------
        res: dict
            Dictionary containing the quantities computed by the model:
            - `heart_rate`: forecasted heart rate (batch_size, sequence_length)
            - `demand`: forecasted demand (batch_size, sequence_length)
            - `intensity`: forecasted intensity (batch_size, sequence_length)
            - `fatigue_coefficient`: forecasted fatigue coefficient (batch_size, sequence_length)
            - `weather_coefficient`: forecasted weather coefficient (batch_size, sequence_length)
            - `ode_params`:
                Dictionary containing the parameters of the ODE: A, B, alpha, beta, hr_min, hr_max (batch_size)
            - `initial_heart_rate_and_demand`: initial heart rate and demand (batch_size, 2)
        """
        self.initialize_batch(
            activity,
            times,
            workout_id,
            subject_id,
            history,
            history_length,
            weather,
        )

        x0 = self._initial_heart_rate_and_demand
        times = torch.arange(0, 1 + times.shape[1]).float()
        xs = odeint(
            self,
            x0,
            times,
            method="rk4",
            options=dict(step_size=step_size),
        )

        res = {
            "heart_rate": xs[1:, :, 0].transpose(0, 1),
            "demand": xs[1:, :, 1].transpose(0, 1),
            "intensity": self._intensity,
            "fatigue_coefficient": self._fatigue_coefficient,
            "weather_coefficient": self._weather_coefficient,
            "ode_params": self._ode_params,
            "initial_heart_rate_and_demand": self._initial_heart_rate_and_demand,
            "workout_embedding": self._workout_embeddings,
        }
        return res

    def forecast_single_workout(
        self,
        workout,
        step_size=1.0,
        convert_to_numpy=True,
    ):
        """
        Convenience function to forecast a single workout, in the format of the WorkoutDataset.
        """
        from ode.data import workout_dataset_collate_fn

        workout = workout_dataset_collate_fn([workout])

        res = self.forecast_batch(
            workout["activity"],
            workout["time"],
            workout["workout_id"],
            workout["subject_id"],
            workout["history"],
            workout["history_length"],
            workout["weather"],
            step_size=step_size,
        )
        for k in res:
            if type(res[k]) == torch.Tensor:
                res[k] = res[k].squeeze(0)
                if convert_to_numpy:
                    res[k] = res[k].detach().numpy()
            if type(res[k]) == dict:
                for k2 in res[k]:
                    res[k][k2] = res[k][k2].squeeze(0)
                    if convert_to_numpy:
                        res[k][k2] = res[k][k2].detach().numpy()
        return res
