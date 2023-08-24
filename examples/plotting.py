#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_workout_predictions(model, workout):
    """
    Large plot of a single workout.
    The plot is divided into two subplots:
    - the top subplot shows the true heart rate and the predicted heart rate, as well as the demand.
    - the bottom subplot shows the different activity measurements.
    Both plot show the time in the x-axis.
    The title of the plot show the ODE parameters for the workout.

    """

    time_in_datetime = pd.to_datetime(workout["time"], unit="s")
    predictions = model.forecast_single_workout(workout)
    predictions_hr = predictions["heart_rate"]
    predictions_demand = predictions["demand"]

    # title with ODE parameters
    ode_params = predictions["ode_params"]
    title = f"ODE parameters: "
    for k, v in ode_params.items():
        title += f"{k}={v:.2f}, "
    title = title[:-2]

    # plot the heart rate and demand
    fig, ax = plt.subplots(2, 1, figsize=(20, 8), sharex="all")
    fig.suptitle(title)

    ax_top = ax[0]
    ax_top.plot(
        time_in_datetime,
        workout["heart_rate"],
        color="#ee2a15",
        ls=(0, (1, 1)),
        label="True HR",
    )
    ax_top.plot(
        time_in_datetime, predictions_hr, color="#ee2a15", label="Predicted HR (bpm)"
    )
    ax_top.set_ylabel("Heart rate (bpm)")
    ax_top.plot(
        time_in_datetime, predictions_demand, color="#1f77b4", label="Predicted demand"
    )
    ax_top.set_ylim(50, 200)
    ax_top.legend(loc="lower right")

    ax_bottom = ax[1]
    # plot the measurements. Each measurement is a different color, and has its own twin y-axis.
    # Each y-axis has the name of the measurement as its label.
    legend_handles, legend_labels = [], []
    for i, measurement_name in enumerate(workout["activity_measurements_names"]):
        if i == 0:
            ax_bottom_twin = ax_bottom
        else:
            ax_bottom_twin = ax_bottom.twinx()
        ax_bottom_twin.set_ylabel(measurement_name)
        ax_bottom_twin.tick_params(axis="y", labelcolor=f"C{i}")
        ax_bottom_twin.plot(
            time_in_datetime,
            workout["activity"][:, i],
            color=f"C{i}",
            label=measurement_name,
        )
        legend_labels.append(measurement_name)
        legend_handles.append(ax_bottom_twin.get_lines()[0])

    # one legend for all the measurements, need to get the handles and labels from the different y-axes
    ax_bottom.legend(legend_handles, legend_labels)

    # format the x-axis to show only the hours and minutes
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax_bottom.set_xlabel("Time")
    ax_bottom.set_xlim(time_in_datetime[0], time_in_datetime[-1])
