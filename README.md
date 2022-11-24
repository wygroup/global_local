# "global + local" strategy
 
The package replicates the training and testing of "global + local" model using TMSACs.

##  Prerequisites

This package requires:

- tensorflow
- scikit-learn
- pymatgen
- networkx
- pickle, joblib
- matplotlib

The easiest way of installing the prerequisites is via [conda](https://www.anaconda.com). 
After installing `conda`, run the following command to create a new environment named `global_local` 
and install all prerequisites:

```bash
conda upgrade conda
conda create --name global_local python=3.8 -c conda-forge
```

This creates a conda environment for running global_local. 

Before using global_local, activate the environment by:

```bash
source activate global_local
```

After activating the environment, tensorflow 2.6 for gpu is needed:

```bash
conda install tensorflow-gpu=2.6 cudatoolkit=11.3 cudnn=8.2
```

The above three versions should be compatible with each other.

Then, in directory `global_local`, you can test if all the prerequisites are installed properly by running:

```bash
python graphs.py
python pixel.py
python global_local.py
```

`graph.py` and `pixel.py` generate descriptor and grid inputs named `graphs.pkl` and `pixels.pkl` in current folder. 
`global_local.py` train and test CNN model using generated inputs.

`E_OH_all.csv` will be generated after model training and testing, which containing predicted and true values.
The log of the model run is recorded in the `log` folder. 
The best model in the training process is saved in the `model_opt` folder. 

`42`, `24` and `22` denote different cell expansion coefficients. 
The distribution of N outside defects in `42_2` is different from that in `42`. 
The structure of the `42` should be:

```
global_local
├─ 42                     # cell expansion coefficient
│  ├─ 0N                  # N content outside defect
│  │  ├─ SV               # defect containing 0 N
│  │  │  ├─ Sc            # TM atom
│  │  │  │      POSCAR
│  │  │  │      POTCAR
│  │  │  ├─ Ti
│  │  │  │      POSCAR
│  │  │  │      POTCAR
│  │  │  ├─ ...
│  │  ├─ SV_1N            # defect containing 1 N
│  │  │  ├─ Sc
│  │  │  │      POSCAR
│  │  │  │      POTCAR
│  │  │  ├─ ...
```

`22`, `24` and `42_2` folders also have the same structure as `42`.
