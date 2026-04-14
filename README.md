# nlp-predicting-politics
Predicting Politics and Ideology Final Project with Jack Houck, Brady Johnsen, and Jack Timmermans

# TODO:
- document data sources, processing done, etc.
    - https://www.presidency.ucsb.edu/documents/app-categories/written-presidential-orders/presidential/executive-orders?items_per_page=60
    - https://www.govinfo.gov/bulkdata/BILLS

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


