# nlp-predicting-politics

Predicting Politics and Ideology Final Project with Jack Houck, Brady Johnsen, and Jack Timmermans.

## Directory Structure

The repo contains 3 main experiment directories: BERT, deep_learning, and statical modeling. These contain the scripts corresponding to the type of model used for classication: BERT is distilBERT and legalBERT, deep_learning is the CNN, and staticial modeling is the bag of words, tfidf, and word2vec models. 

The eo_data directory contains the full dataset of executive orders, including the cleaned versions, the test/training split, and the modern presidents dataset.

The scripts directory contains miscellanous scripts, namely the initial baselines script, the webscraping/dataset script, and the named-entity recognition script(s).

The templates directory includes a single template file initially used for consistency of constructing the dataset.

The word analysis directory contains the code used to do term frequency analysis of the text in the dataset. 

## Data Collection

We have two datasets for US Presidential executive orders and Congressional bills found in `clean_data` and `unclean_data`, for processed and unprocessed data respectively.

The executive order corpus consists of one text file of every executive order, spanning 200 years, organized by president, dating back to John Quincy Adams's announcement of the deaths of Thomas Jefferson and John Adams in 1826 all the way to Donald Trump's latest order on April 03, 2026. There are 10818 orders within this corpus. The orders can be found at https://www.presidency.ucsb.edu/documents/app-categories/written-presidential-orders/presidential/executive-orders.

The Congressional bill corpus consists of one text file per bill from GovInfo Bulk Data for the 113th through 119th Congresses (2013-2026). It includes 103,905 bills introduced in both the House and Senate. The data can be found at https://www.govinfo.gov/bulkdata/BILLS.

## How to run code

Working from the cloned repo, each script should be able to be run via its main function with no conflicts. If errors occur, it is likely due to changes in the organization of the repo overtime, which may require copying the correct relative path to the dataset split directories.

## Non-standard Libraries or APIS

## Links to models


## Links to notebooks/tutorials

explain the directory structure
where to look for code
where to look for data files
how I would run the code if I wanted to (I won't run it unless it looks totally wonky)
what order to run the programs in (if applicable)
links to where you got the data 
links to any non-standard libraries or APIs you used (e.g., don't tell me about scikit or pandas, but do tell me about the Genius API or the Mistral library)
links to where you found any models you used
links to notebooks or tutorials you used that I did not provide


# Setup
1) Clone the repo
```bash
  git clone https://github.com/b-johnsen/nlp-predicting-politics.git
```
2) Install dependencies (use conda env or venv if preferred)
```bash
  pip install -r requirements.txt
```

3) Download the spacy language model (needed for NER)
```bash
  python -m spacy download en_core_web_sm
```


