# nlp-predicting-politics
Predicting Politics and Ideology Final Project with Jack Houck, Brady Johnsen, and Jack Timmermans

# TODO:
- do all executive orders: https://www.presidency.ucsb.edu/documents/app-categories/written-presidential-orders/presidential/executive-orders?items_per_page=60
- take in congressional data: https://www.govinfo.gov/bulkdata/BILLS
- document data sources, processing done, etc.

# Setup
1) Clone the repo
  git clone https://github.com/b-johnsen/nlp-predicting-politics.git
2) Install dependencies
  pip install -r requirements.txt
  (or use conda env)
3) Download the spacy language model
  python -m spacy download en_core_web_sm
  (needed for NER)

