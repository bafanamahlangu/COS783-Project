# Digital Forensics AI — Keyword Search Tool

A university project that uses AI to search through forensic text data like emails, messages, and logs.

Instead of looking for exact words, it understands the **meaning** behind your search. For example, searching for "drug transaction" will also find messages like "send the money for the drugs" or "wire the cash".

---
## Team Members

- Tedy Mbusi Dube - u26812976
- Khensani Emmelda Chabalala - u23826305
- Bafana Christopher Mahlangu - u26812453


---

## What it does

- Takes a CSV file of messages as input
- Takes a search query from the user
- Returns the top 5 most relevant messages ranked by similarity score
- Highlights named entities like people, places, and dates found in the results

---

## Project structure

```
COS783-Project-main/
├── Data/
│   ├── example.csv                      # Sample data file
│   └── fraud_emails.csv                 # Demonstration data file
├── file_reader.py                        # Reads and splits the CSV into sentences
├── keyword_searcher.py                   # AI search logic (embeddings + similarity)
├── main.py                               # CLI entry point
├── app.py                                # Streamlit web app
├── fraud_email_data_creation.ipynb       # Dataset creation notebook
└── requirements.txt                      # Python dependencies
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Download the English language model for spaCy**
```bash
python -m spacy download en_core_web_lg
```

---

## How to run

```bash
python main.py
```

You will be asked two questions:
1. The name of your CSV file (e.g. `example.csv`)
2. The keywords you want to search for (e.g. `drug transaction`)

---

## CSV file format

Your data file must have exactly 2 columns — a unique ID and a message:

```
message_id,message
1,We are live bro. Send the money for the drugs
2,Please wire the cash to Jamie
```

---

## Example output

```
Top 5 results for: 'drug transaction'
--------------------------------------------------
[1] Score: 0.5995 | Message ID: 2
    Text: Send the money for the drugs

[2] Score: 0.3502 | Message ID: 3
    Text: Please wire the cash to Jamie to wrap up yesterday's transaction
    Entities: {'PERSON': ['Jamie'], 'DATE': ['yesterday']}
```
## Running the web app

```bash
streamlit run app.py
```

Or if that doesn't work:

```bash
python -m streamlit run app.py
```

The web app lets you upload a CSV, set the number of results, filter by similarity score, and export results.

---

## Built with

- [Sentence Transformers](https://www.sbert.net/) — AI model for understanding text meaning
- [spaCy](https://spacy.io/) — Named entity recognition (people, places, dates)
- [PyTorch](https://pytorch.org/) — Backend for running the AI model
- [pandas](https://pandas.pydata.org/) — Reading CSV files
- [NLTK](https://www.nltk.org/) — Splitting messages into sentences
- [Streamlit](https://streamlit.io/) — Web app interface
