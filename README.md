# Thesis_Ellinor

Based on news data that is labeled fake and real it will produce a style based classification and a matrix with  precision, recall and f1-score. Part of my thesis "#FakeNews- Validering av sanningshalten i nyhetsströmmar"

Make sure the following files are present as per the directory structure before running the code,

Prerequisites:
FakeNewsNet dataset is downloaded. Can be found here, https://github.com/KaiDMML/FakeNewsNet.git

Create working folders "Resultat" and "nltk_prep_result" both should have subfolders "Fake" and "Real"
Adjust folder paths in code to match.


# Dependencies

The code runs on the Intel(R) Core(TM) i7-10710U CPU @ 1.10GHz 1.61 GHz, WSL Ubuntu 20.04 LTS On Windows 10 Business 32 Gb RAM with the following packages installed:
```
python 3.8.10
torch 1.3.1
nltk 3.7
numpy 1.23.4
pandas 1.5.1
matplotlib 3.6.1
scikit_learn 1.1.3
trainer 0.0.16
```



# Run

 
```
python3 main.py
