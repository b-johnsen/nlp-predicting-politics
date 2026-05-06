# nlp-predicting-politics

Predicting Politics and Ideology Final Project with Jack Houck, Brady Johnsen, and Jack Timmermans.


## Directory Structure

The repo contains 3 main experiment directories: BERT, deep_learning, and statical modeling. These contain the scripts corresponding to the type of model used for classication: BERT has distilBERT and legalBERT models, deep_learning is the CNN, and staticial modeling is the bag of words, tfidf, and word2vec models. 

The eo_data directory contains the full dataset of executive orders, including the cleaned versions, the test/training split, and the modern presidents dataset, including scripts to web scrape the data and clean/split the data.

Tbe bills_data directory contains a compressed file of all bills from the 113th through 119th Congresses, as well as the scripts to retrieve the data and clean/split the data.

The NER directory contains all scripts to mask certain entities in the bills/EO datasets using named-entity recognition.

The baselines directory contains the script to run the most_frequent, stratified, and uniform baselines.

The templates directory includes a single template file initially used for consistency of constructing the dataset.

The word analysis directory contains the code used to do term frequency analysis of the text in the dataset. 

The eo_count_analysis directory contains the scripts to create visualizations of the EO counts over time.


## Data Collection

We have two datasets for US Presidential executive orders and Congressional bills found in `clean_data` and `unclean_data`, for processed and unprocessed data respectively.

The executive order corpus consists of one text file of every executive order, spanning 200 years, organized by president, dating back to John Quincy Adams's announcement of the deaths of Thomas Jefferson and John Adams in 1826 all the way to Donald Trump's latest order on April 03, 2026. There are 10818 orders within this corpus. The orders can be found at https://www.presidency.ucsb.edu/documents/app-categories/written-presidential-orders/presidential/executive-orders.

The Congressional bill corpus consists of one text file per bill from GovInfo Bulk Data for the 113th through 119th Congresses (2013-2026). It includes 103,905 bills introduced in both the House and Senate. The data can be found at https://www.govinfo.gov/bulkdata/BILLS.


## How to run code

Provided the dependencies are installed from `requirements.txt`, each script should be able to be run via its main function with no conflicts. Ensure that the path variables are correct for your configuration.  

For NER, be sure to install the relevant spacy model using `python -m spacy download en_core_web_sm`.


## Links to models

* google news word2vec: https://huggingface.co/fse/word2vec-google-news-300
* legalBERT: https://huggingface.co/nlpaueb/legal-bert-base-uncased


## Links to notebooks/tutorials

* tf-idf wordcloud: https://ayselaydin.medium.com/6-creating-a-word-cloud-using-tf-idf-in-python-2554742d86d9
* tf-idf vectorizer: https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/03-TF-IDF-Scikit-Learn.html
* bow vs tf-idf: https://www.analyticsvidhya.com/blog/2021/07/bag-of-words-vs-tfidf-vectorization-a-hands-on-tutorial/


## ITEMS NEEDED (DELETE LATER)
* explain the directory structure
* where to look for code
* where to look for data files
* how I would run the code if I wanted to (I won't run it unless it looks totally wonky)
* what order to run the programs in (if applicable)
* links to where you got the data
* links to any non-standard libraries or APIs you used (e.g., don't tell me about scikit or pandas, but do tell me about the Genius API or the Mistral library)
* links to where you found any models you used
* links to notebooks or tutorials you used that I did not provide
