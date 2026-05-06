import spacy 
from pathlib import Path

model = spacy.load("en_core_web_sm")

# Uncomment to see list of labels
# print(model.pipe_labels['ner'])

def remove_entities(text, labels_to_remove={"PERSON", "GPE", "DATE", "ORG", "NORP"}):
    ''' PERSON - names; GPE - states, countries, cities;
        DATE - times; NORP - nationalities, religion/politics;
        ORG - might remove govt agencies?'''
    doc = model(text)
    clean = text
    for ent in reversed(doc.ents):
        if ent.label_ in labels_to_remove:
            clean = clean[:ent.start_char] + " " + clean[ent.end_char:]
    return clean

# This is if you are in the /scripts directory running the program
input_dir = Path("../unclean_data/all_executive_orders_txt_clean/")
output_dir = Path("../clean_data/clean_eo/")

for txt_file in input_dir.rglob("*.txt"):
    text = txt_file.read_text(encoding="utf-8")
    cleaned = remove_entities(text)
    
    # recreate the same folder structure in the output
    out_path = output_dir / txt_file.relative_to(input_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(cleaned, encoding="utf-8")
    
    print(f"Cleaned: {txt_file}")

print("Done!")