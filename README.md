Code for the ...xxx paper xxx


Make sure the following files are present as per the directory structure before running the code,
```
StyleBased
├── README.md
├── *.py
└───data
    └── <dataset name> 
        ├── input (for input data in csv-files or txt-files)
        │   ├── *.csv
        │   └── */*.txt
        ├── workload
        │   ├── Fake
        │   └── Real
        └── result_nltk_prep
            ├── Fake
            └── Real

```


# Dependencies

Our code runs on the GeForce RTX 2080 Ti (11GB), with the following packages installed:
```
python 3.7
torch 1.3.1
nltk 3.2.5
tqdm
numpy
pandas
matplotlib
scikit_learn
xlrd (pip install xlrd)
```


