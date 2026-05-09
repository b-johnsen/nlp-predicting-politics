# nlp-predicting-politics

Predicting Politics and Ideology Final Project with Jack Houck, Brady Johnsen, and Jack Timmermans.


## Directory Structure

The repository contains 4 main directories that contain relevant subdirectories.

data directory:
- The eo_data directory contains the full dataset of executive orders, including the cleaned versions, the test/training split, and the modern presidents dataset, including scripts to web scrape the data and clean/split the data.
- Tbe bills_data directory contains a compressed file of all bills from the 113th through 119th Congresses, as well as the scripts to retrieve the data and clean/split the data.

data_analysis directory:
- The word analysis directory contains the code used to do term frequency analysis of the text in the dataset. 
- The eo_count_analysis directory contains the scripts to create visualizations of the EO counts over time.

models directory:
- The baselines directory contains the script to run the most_frequent, stratified, and uniform baselines.
- The BERT directory contains scripts to train the distilBERT and legalBERT models, as well as a script to test them for each president.
- The deep_learning directory contains scripts to train the CNN and create its embeddings.
- The statistical modeling directory contains the scripts to train the bag of words, tfidf, and word2vec models. 

resources directory:
- The NER directory contains all scripts to mask certain entities in the bills/EO datasets using named-entity recognition.
- The templates directory includes a single template file initially used for consistency of constructing the dataset.


## Data Collection

We have two datasets for US Presidential executive orders and Congressional bills found in the `data` directory.

The executive order corpus consists of one text file of every executive order, spanning 200 years, organized by president, dating back to John Quincy Adams's announcement of the deaths of Thomas Jefferson and John Adams in 1826 all the way to Donald Trump's latest order on April 03, 2026. There are 10818 orders within this corpus. The orders can be found at https://www.presidency.ucsb.edu/documents/app-categories/written-presidential-orders/presidential/executive-orders.

The Congressional bill corpus consists of one text file per bill from GovInfo Bulk Data for the 113th through 119th Congresses (2013-2026). It includes 103,905 bills introduced in both the House and Senate. The data can be found at https://www.govinfo.gov/bulkdata/BILLS.


## How to run code

Provided the dependencies are installed from `requirements.txt`, each script should be able to be run via its main function with no conflicts. Ensure that the path variables are correct for your configuration.  

For NER, be sure to install the relevant spacy model using `python -m spacy download en_core_web_sm`.  

Since the `bills_data.zip` file is so large, you may need to install [Git LFS](https://git-lfs.com) to interact with the repository.


## Links to models

* google news word2vec: https://huggingface.co/fse/word2vec-google-news-300
* legalBERT: https://huggingface.co/nlpaueb/legal-bert-base-uncased


## Links to notebooks/tutorials

* tf-idf wordcloud: https://ayselaydin.medium.com/6-creating-a-word-cloud-using-tf-idf-in-python-2554742d86d9
* tf-idf vectorizer: https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html
* bow vs tf-idf: https://www.analyticsvidhya.com/blog/2021/07/bag-of-words-vs-tfidf-vectorization-a-hands-on-tutorial/
