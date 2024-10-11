import spacy
import re
from typing import List, Dict

def extract_risk_statements_from_10k(file_path: str) -> List[Dict[str, str]]:
    # Load the English NLP model
    nlp = spacy.load("en_core_web_sm")
    
    # Read the 10-K filing
    with open(file_path, 'r') as file:
        text = file.read()
    
    # Find the "Risk Factors" section
    risk_factors_section = re.search(r"Item 1A.\s*Risk Factors(.*?)Item 1B", text, re.DOTALL)
    if risk_factors_section:
        risk_factors_text = risk_factors_section.group(1)
    else:
        return []
    
    # Process the text with spaCy
    doc = nlp(risk_factors_text)
    
    # Extract risk statements
    risk_statements = []
    for sent in doc.sents:
        if any(token.text.lower() in ["risk", "uncertainty", "could", "may"] for token in sent):
            risk_statements.append({
                "text": sent.text,
                "entities": [(ent.text, ent.label_) for ent in sent.ents]
            })
    
    return risk_statements