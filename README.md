# Modeling Personalized Heart Rate Response to Exercise and Environmental Factors with Wearables Data

This repository contains the code associated with the research paper _Modeling
Personalized Heart Rate Response to Exercise and Environmental Factors with Wearables
Data_ (Nazaret et al. NPJ Digital Medicine, 2023).
[Link forthcoming](https://www.nature.com/npjdigitalmed).

## Abstract
Heart rate (HR) response to workout intensity reflects fitness and
cardiorespiratory health. Physiological models have been developed to describe
such heart rate dynamics and characterize cardiorespiratory fitness. However,
these models have been limited to small studies in controlled lab environments
and are challenging to apply to noisy — but ubiquitous — data from wearables.
We propose a hybrid approach that combines a physiological model with flexible
neural network components to learn a personalized, multidimensional
representation of fitness. The physiological model describes the evolution of
heart rate during exercise using ordinary differential equations (ODEs). ODE
parameters are dynamically derived via a neural network connecting personalized
representations to external environmental factors, from area topography to
weather and instantaneous workout intensity. Our approach efficiently fits the
hybrid model to a large set of 270,707 workouts collected from wearables of
7,465 users from the Apple Heart and Movement Study. The resulting model
produces fitness representations that accurately predict full HR response to
exercise intensity in future workouts, with a per workout median error of 6.1
BPM [4.4 - 8.8 IQR]. We further demonstrate that the learned representations
correlate with traditional metrics of cardiorespiratory fitness, such as VO2
max (explained variance 0.81 ± 0.003). Lastly, we illustrate how our model is
naturally interpretable and explicitly describes the effects of environmental
factors such as temperature and humidity on heart rate, e.g., high temperatures
can increase heart rate by 10%. Combining physiological ODEs with flexible
neural networks can yield interpretable, robust, and expressive models for
health applications.

## Getting Started

These instructions will help you get a copy of the hybrid ODE model up and running on your local machine 
for development and testing purposes. See the [Usage](#usage) section for notes on how to use the code.

### Installation

1. Clone this repository: `git clone https://github.com/Apple/repo-name.git`
2. Change to the project directory: `cd repo-name`
3. Install the required packages: `pip install -r requirements.txt`

### Dataset
The Apple Heart and Movement Study dataset is not publicly available for privacy reasons.
Users can instead use their own data or other publicly available datasets.

For example, the paper
[_Modeling heart rate and activity data for personalized fitness recommendation_](https://dl.acm.org/doi/pdf/10.1145/3308558.3313643) [1]
provides [a dataset](https://sites.google.com/eng.ucsd.edu/fitrec-project/home) of user sport records collected from wearables
One can use the `examples/preprocess.py` script to produce the data format required
for our algorithm, e.g.,
```
python examples/preprocess.py --input_path /path/to/dataset_folder --output_path /output/path
```
which will generate a file `endomondo.feather` in the output folder.

### Code Structure

The code is organized as follows:

```
.
├── ode
│   ├── data.py                 # Data loading and preprocessing
│   ├── modules_cnn.py          # Modules for the CNN encoder
│   ├── modules_dense_nn.py     # Modules for all scalar dense NNs
│   ├── ode.py                  # Hybrid ODE model
│   └── trainer.py              # Trainer for the ODE model
├── examples
│   ├── preprocess.py            # Script to preprocess the endomondo dataset
│   ├── plotting.py             # Helper functions to plot the data
│   └── train_ode_model.ipynb   # Example notebook to train the ODE model
├── readme.md                   # This file
└── requirements.txt            # Required packages
```

### Usage

* **Step 1: Prepare the data.**
The `WorkoutDataset` class in `ode/data.py` loads the data from a pandas dataframe and preprocesses it, according to 
the `WorkoutDatasetConfig` class. In the current implementation, all the measurements (HR, speed ...) are assumed to be sampled on a
uniformly sampled time grid (that we can obtain by interpolating the data).
The following data is required in the dataframe:
  * the user id that performed the workout
  * the id of the workout
  * the time grid on which the heart rate and the activity are measured
  * a list of heart rate measurements 
  * a list of heart rate measurements normalized in a reasonable range for neural networks
  * one column per activity measurements (e.g. list of horizontal speeds, vertical speed, etc.)
  * [optional] one column per weather measurement (e.g. one temperature, humidity, etc.)


* **Step 2: Define the ODE model.**
  * The `ODEConfig` class specifies the hyperparameters of the `ODEModel`.


* **Step 3: Train the ODE model.**
  * The function `train_ode_model` in `ode/trainer.py` trains the `ODEModel`.

An example is provided in `examples/train_ode_model.ipynb`. 


## Authors

* **Achille Nazaret** - *Columbia University* (Work done while at Apple)
* **Sana Tonekaboni** - *University of Toronto* (Work done while at Apple)
* **Greg Darnell** - *Apple*
* **Shirley Ren** - *Apple*
* **Guillermo Sapiro** - *Apple*
* **Andrew C. Miller** - *Apple*

## Citation

If you use this code or dataset in your research or publications, please cite
the original research paper:

```
Nazaret, A., Tonekaboni, S., Darnell, G., Ren, S., Sapiro, G., & Miller, A.  Modeling
Personalized Heart Rate Response to Exercise and Environmental Factors with Wearables
Data.  NPJ Digital Medicine, 2023.
```

Bibtex entry
```
@article{nazaret2023modeling,
  title={Modeling Personalized Heart Rate Response to Exercise and Environmental Factors with Wearables Data},
  author={Nazaret, Achille and Tonekaboni, Sana and Darnell, Gregory and Ren, Shirley and Sapiro, Guillermo and Miller, Andrew C.},
  journal={NPJ Digital Medicine},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## References

[1] Ni, Jianmo, Larry Muhlstein, and Julian McAuley.
"Modeling heart rate and activity data for personalized fitness recommendation." The World Wide Web Conference. 2019.
