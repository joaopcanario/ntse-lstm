# NTSE-LSTM: Noise-tolerant Self-embedded LSTM

Source code of the paper entitled as "Noise-tolerant Self-embedded LSTM for Seismic Event Classification", published on [IEEE MLSP 2023](https://2023.ieeemlsp.org/).

## Steps to Project Execution

This code was developed using **Python 3.8**, and you can follow the steps below for project execution:

1. Create a Python virtual environment
2. Install project requirements `pip install -r requirements.txt`
3. Create a `data/` folder, download and save the experiment dataset (originally published on [https://doi.org/10.1016/j.dib.2020.105627](https://doi.org/10.1016/j.dib.2020.105627)).
4. Split the dataset into train and test sets to save volcano data and labels using the [NPY format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html). We strongly recommend using the `x_y_trainset.npy` and `x_y_testset.npy` names to save the train and test datasets on the data folder.

After doing all the steps, you can run `python main.py`.
