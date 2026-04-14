# nlp-predicting-politics

Predicting Politics and Ideology Final Project with Jack Houck, Brady Johnsen, and Jack Timmermans.


## Data Collection

We have two datasets for US Presidential executive orders and Congressional bills found in `clean_data` and `unclean_data`, for processed and unprocessed data respectively.

The executive order corpus consists of one text file of every executive order, spanning 200 years, organized by president, dating back to John Quincy Adams's announcement of the deaths of Thomas Jefferson and John Adams in 1826 all the way to Donald Trump's latest order on April 03, 2026. There are 10818 orders within this corpus. The orders can be found at https://www.presidency.ucsb.edu/documents/app-categories/written-presidential-orders/presidential/executive-orders.

The Congressional bill corpus consists of one text file per bill from GovInfo Bulk Data for the 113th through 119th Congresses (2013-2026). It includes all bills introduced in both the House and Senate. The data can be found at https://www.govinfo.gov/bulkdata/BILLS.


## Data Processing

TODO: Describe data processing steps


# TODO:

- Jack H: Finish NER
- Jack T: Data script
- Jack T: word map
- Jack T: Statistical models graphs, word2vec
- Jack H: CNN
- Brady: DistillBERT
- Jack H: Tranformer (Huggingface)


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


