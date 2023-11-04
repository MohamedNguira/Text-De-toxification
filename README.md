# Text-De-toxification
## Practical Machine Learning and Deep learning course
### By Mohamed Nguira
### BS21-RO
### m.nguira@innopolis.university

## 1) Structure of the repo:
```
text-de-toxification
├── README.md # The top-level README
│
├── data 
│   ├── external # Data from third party sources
│   ├── interim  # Intermediate data that has been transformed.
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│
├── notebooks  #Contains all used notebooks for the assignment  
│ 
├── references   # Data dictionaries, manuals, and all other explanatory materials.
│
├── reports      # Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures  # Generated graphics and figures to be used in reporting
│
├── requirements.txt # The requirements file for reproducing the analysis environment, e.g.
│                      generated with pip freeze › requirements. txt'
└── src                 # Source code for use in this assignment
    │                 
    ├── data            # Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
    │   ├── predict_model.py
    │   └── train_model.py

```

## 2) How to use this repo:
### a) Download the necessary data:
Github has a limitation on the file size limit that I can upload. Therefore, in order to use this repo, you first need to clone it and then run this command inside the "src\data" folder:
```
python download.py
```

Basically just run the download script. This will download 2 important materials:

-All the data used in the project (Internal data, interprocessed data, final data and external data)

-Checkpoints achieved after training the models used in this assignment.

Note: All of these will be saved in their respective folder mentioned in the assignment description

### b) Preprocess the data:
I have 2 main notebooks for preprocessing the data, each of them is responsible for preparing the data before giving it to the model to start the training. There 2 notebooks are:
```
1.0-preprocessing.ipynb

1.1-preprocessing2.ipynb

```

Note: Another side notebook (1.2-analysis.ipynb)was used for analysing some data characterics

### c) Training models:
There are a total of 3 models which needs to be trained for this assignment, more details about them can be found inside the report folder.
To train them you just need to run the code in the following notebooks:

```
2.0-model1.ipynb

2.1-model2.ipynb

```

### d) Testing the algorithm:
There are multiple ways to test the algorithm. Note that first thing to be done is to download the checkpoints mentioned in step (a)

First of all you can create you own test data set and put it inside the "data\external" folder and then run the code present in the following notebook:
```
3.0-predict.ipynb
```

You can all also run the script found in "src\models\predict\predict.py" using the command:

```
python predict.py
```

Both of these ways will generated a .txt file in their respective directory containing results of the prediction in the same order given in the input.

### e) Results and evaluation:
For this part, check the 2nd report found in reports folder