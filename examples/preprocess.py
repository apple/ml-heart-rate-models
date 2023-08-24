#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os

import numpy as np
import pandas as pd
import tqdm

tqdm.tqdm.pandas()


def load_data(input_path: str):
    """Load the endomondo data from the json file and return a pandas dataframe."""
    lines = []
    with open(input_path, "r") as f:
        print_cooldown = True
        for line in f:
            # only keep the lines with the sport "run" in them
            if "run" in line:
                # read json line, and fix the quotes in the json
                lines.append(json.loads(line.replace("'", '"')))
                print_cooldown = False
            # show progress
            if len(lines) % 10_000 == 0 and not print_cooldown:
                print(f"Loaded {len(lines)} running workouts so far.")
                print_cooldown = True

    df = pd.DataFrame(lines)
    return df


def haversine_distances(longitudes, latitudes):
    """Compute the distances between consecutive locations from a list of longitudes and latitudes."""
    earth_radius = 6_371_000
    phis = np.radians(latitudes)
    delta_phi = np.radians(np.diff(latitudes))
    delta_lambda = np.radians(np.diff(longitudes))

    a = (
        np.sin(delta_phi / 2.0) ** 2
        + np.cos(phis[:-1]) * np.cos(phis[1:]) * np.sin(delta_lambda / 2.0) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = c * earth_radius
    return np.cumsum(np.insert(distances, 0, 0))


def interpolate(
    time,
    values,
    target_grid,
    make_cumulative: bool = False,
    remove_offset: bool = False,
    max_consecutive_nan: int = 5,
):
    """Interpolate the values to the target grid.

    Parameters
    ----------
    time : array-like
        The time points of the values.
    values : array-like
        The values to interpolate.
    target_grid : array-like
        The target grid to interpolate to.
    make_cumulative : bool, optional
        Whether to make the values cumulative, by default False.
    remove_offset : bool, optional
        Whether to remove the offset of the values, by default False.
    max_consecutive_nan : int, optional
        The maximum number of consecutive NaNs to allow, by default 5.

    """

    if values is None:
        return None

    target_grid = pd.DataFrame(index=target_grid)

    source_df = pd.DataFrame(index=time, columns=["values"], data=values).dropna()
    if make_cumulative:
        source_df["values"] = source_df["values"].cumsum()

    if not source_df.index.is_monotonic:
        # happens when some people decide to start a workout during a DST change, we just ignore these workouts
        return None

    # smooth/interpolate the values
    # source_df = source_df.rolling(f"{interval}S", min_periods=1).median()

    target_df = pd.concat(
        [target_grid, source_df],
        axis=0,
    ).sort_index()

    # interpolate the values with a limit on the number of consecutive NaNs
    target_df.interpolate(
        "time", inplace=True, limit=max_consecutive_nan, limit_area="inside"
    )
    target_df = target_df[~target_df.index.duplicated(keep="first")]
    target_values = target_df.reindex(target_grid.index)

    if target_values["values"].isnull().any():
        return None

    if remove_offset and len(target_values):
        index = target_values["values"].first_valid_index()
        if index is None:
            return None
        target_values["values"] -= target_values["values"][index]

    return target_values["values"].values.tolist()


def plot_endomondo_workout(workout):
    """Plot a workout from the Endomondo dataset."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    ax[0].plot(workout["time_grid"], workout["heart_rate"], color="red")
    ax[0].set_ylabel("Heart rate (bpm)")

    ax[1].plot(workout["time_grid"], workout["speed_h"] * 3.6)
    ax[1].set_ylabel("Speed (km/h)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylim(0, 30)
    # twin axis
    ax2 = ax[1].twinx()
    ax2.plot(workout["time_grid"], workout["speed_v"], color="green")
    ax2.set_ylabel("Vertical speed (m/s)")
    ax2.set_ylim(-5, 5)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the raw Endomondo dataset: `endomondoHR.json`",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Output directory of the processed data: `endomondo.feather`",
    )
    args = parser.parse_args()

    df = load_data(os.path.join(args.input_path, "endomondoHR.json"))

    # select only the runs
    df = df[df["sport"] == "run"]

    # convert the times to datetime and select only the workouts that are between 15 minutes and 2 hours
    df["timestamp_dt"] = df.timestamp.apply(
        lambda a: np.array(a, dtype="datetime64[s]")
    )
    df["start_dt"] = df["timestamp_dt"].str[0]
    df["end_dt"] = df["timestamp_dt"].str[-1]
    df["duration"] = df["end_dt"] - df["start_dt"]
    df = df[df["duration"].dt.total_seconds().between(15 * 60, 2 * 60 * 60)]

    # remove duplicates
    df.drop_duplicates(["userId", "start_dt"], inplace=True)
    df.dropna(subset=["latitude", "longitude", "altitude", "heart_rate"], inplace=True)
    df = df[df["heart_rate"].apply(min) > 45]
    df = df[df["heart_rate"].apply(max) < 215]

    grid_interval = 10
    df["time_grid"] = df.progress_apply(
        lambda row: pd.date_range(
            row["start_dt"] + pd.Timedelta(1, "s"),
            row["end_dt"],
            freq=f"{grid_interval}s",
        ).values,
        axis=1,
    )

    columns_to_interpolate = ["latitude", "longitude", "altitude", "heart_rate"]
    for c in columns_to_interpolate:
        df[c] = df.progress_apply(
            lambda row: interpolate(row["timestamp_dt"], row[c], row["time_grid"]),
            axis=1,
        )
    df.dropna(subset=columns_to_interpolate, inplace=True)

    df["distance"] = df.progress_apply(
        lambda row: haversine_distances(row["longitude"], row["latitude"]),
        axis=1,
    )
    # remove workouts with total distance < 1km
    df["total_distance"] = df["distance"].str[-1]
    df = df[df["total_distance"] >= 1000]

    df["speed_h"] = df.apply(
        lambda row: np.diff(row["distance"])
        / (np.diff(row["time_grid"]).astype(float) / 1e9),
        axis=1,
    )
    df["speed_v"] = df.apply(
        lambda row: np.diff(row["altitude"])
        / (np.diff(row["time_grid"]).astype(float) / 1e9),
        axis=1,
    )
    df["heart_rate"] = df["heart_rate"].str[1:]
    df["time_grid"] = df["time_grid"].str[1:]

    df = df[
        ["time_grid", "heart_rate", "speed_h", "speed_v", "userId", "id", "distance"]
    ]
    df["start_dt"] = df["time_grid"].str[0]
    df["end_dt"] = df["time_grid"].str[-1]
    df["time_grid"] = df["time_grid"].apply(lambda x: x.astype(np.int64) / 1e9)
    df["time_grid"] = df["time_grid"].apply(lambda x: (x - x[0]) / (20 * 60))

    df = df[df["speed_h"].apply(max).between(5 / 3.6, 40 / 3.6)]  # 5km/h to 40km/h
    df = df[
        df["speed_v"].apply(lambda x: np.abs(x).max()).between(0, 20 / 3.6)
    ]  # -20m/s to 20m/s
    df["heart_rate_normalized"] = df["heart_rate"].apply(
        lambda x: (np.array(x) - 142) / 22
    )

    df = df.sort_values("start_dt")
    workouts_by_user = df.groupby("userId")[["id", "start_dt"]].agg(list)
    workouts_by_user["n_workouts"] = workouts_by_user["id"].apply(len)
    workouts_by_user = workouts_by_user[workouts_by_user["n_workouts"].between(10, 200)]
    valid_users = set(workouts_by_user.index)
    df = df[df["userId"].isin(valid_users)]

    # define train-test
    test_proportion = 0.2
    workout_train = workouts_by_user["id"].apply(
        lambda x: x[: int(len(x) * (1 - test_proportion))]
    )
    train_ids = set(np.concatenate(workout_train.values))
    df["in_train"] = df["id"].isin(train_ids)

    # give a unique idx to each subject
    df["subject_idx"] = df["userId"].astype("category").cat.codes

    # save the data as a feather file
    df.reset_index(drop=True).to_feather(
        os.path.join(args.output_path, "endomondo.feather")
    )


if __name__ == "__main__":
    main()
