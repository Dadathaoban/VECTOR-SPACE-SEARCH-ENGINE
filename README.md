# Vector Space Search Engine Using an Inverted Index
Python-based vector space search engine implementing tf-idf weighting, inverted indexing, and cosine similarity for document retrieval.
## Overview
This project implements a **vector space–based search engine** in Python using a previously constructed **inverted index**. The system supports efficient document retrieval and ranking through **tf-idf weighting** and **cosine similarity**, enabling relevance-based search over a document corpus.

The work builds upon earlier indexing phases in which an inverted index was created and enhanced through preprocessing techniques such as stop word removal, token filtering, and Porter stemming. Using this refined index, the current implementation delivers an interactive, query-driven search engine that ranks documents according to their semantic similarity to user queries.

---

## Background and Motivation
Inverted indexes are a foundational data structure in modern information retrieval systems. They enable fast lookup of documents containing specific terms and form the backbone of search engines.

This project extends a basic inverted index by:
- Applying linguistic preprocessing to normalize terms
- Computing **term frequency–inverse document frequency (tf-idf)** weights
- Representing documents and queries in a **vector space model**
- Ranking results using **cosine similarity**

The goal is to demonstrate a complete, end-to-end search engine pipeline consistent with classical information retrieval theory.

---

## Preprocessing and Index Construction
The inverted index used in this project was generated with the following preprocessing rules:

- **Stop word removal** to eliminate high-frequency, low-information terms
- **Token filtering**, including:
  - Ignoring tokens shorter than two characters
  - Excluding tokens consisting solely of digits
  - Ignoring tokens beginning with punctuation
- **Porter stemming** to reduce words to their morphological roots (e.g., *learning*, *learned*, *learns* → *learn*)
- **tf-idf weighting** computed for each unique term–document pair

These steps ensure that the index is compact, normalized, and semantically meaningful.

---

## Vector Space Model and Term Weighting
Each document and query is represented as a vector in a high-dimensional term space using **tf-idf weights**.

The inverse document frequency (IDF) of a term \( t \) is defined as:

\[
\text{idf}(t) = \log \left(\frac{N}{df_t}\right)
\]

where:
- \( N \) is the total number of documents
- \( df_t \) is the number of documents containing term \( t \)

The tf-idf weight of term \( t \) in document \( d \) is:

\[
\text{tf-idf}_{t,d} = \text{tf}_{t,d} \times \text{idf}(t)
\]

This weighting scheme down-weights common terms and emphasizes more discriminative terms.

---

## Cosine Similarity for Document Ranking
Document relevance is measured using **cosine similarity**, which computes the cosine of the angle between the document vector \( \vec{d} \) and the query vector \( \vec{q} \):

\[
\cos(\vec{d}, \vec{q}) = \frac{\vec{d} \cdot \vec{q}}{|\vec{d}| |\vec{q}|}
\]

Cosine similarity normalizes for document length, ensuring fair comparison between documents of different sizes. Higher scores indicate greater relevance.

---

## Search Engine Workflow
The search engine operates as follows:

1. **Query Input**  
   Users enter a query consisting of one or more space-separated terms.

2. **Query Preprocessing**  
   Query terms undergo the same preprocessing steps used during indexing (stop word removal, filtering, stemming).

3. **Query Weight Computation**  
   tf-idf weights are computed for query terms using corpus-derived IDF values.

4. **Candidate Document Retrieval**  
   The inverted index is searched to retrieve documents containing **all query terms**.

5. **Cosine Similarity Calculation**  
   Cosine similarity is computed between the query vector and each candidate document vector.

6. **Ranking and Sorting**  
   Documents are ranked in descending order of cosine similarity.

7. **Result Presentation**  
   The top 20 documents (or fewer if fewer are retrieved) are displayed.

---

## Output Details
For each retrieved document, the system displays:
- Document file name
- Cosine similarity score
- Total number of candidate documents retrieved
- Term weight contributions used in similarity computation
- Simpson algorithm statistics (as provided by system output)

If fewer than 20 documents match the query, only the available results are displayed.

---

## Example Results
The system was tested on a corpus of **570 documents** containing **2,474 unique terms** after preprocessing.  
Sample queries (e.g., *"learn"*) correctly retrieved relevant documents, ignored invalid or absent terms, and ranked results according to cosine similarity.
