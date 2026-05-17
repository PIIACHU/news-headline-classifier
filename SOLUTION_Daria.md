# News Headline Classifier
### Multi-class Text Classification with TF-IDF and Logistic Regression

**Author:** Daria Chernomazova  
**Affiliation:** Southern Federal University, Applied Mathematics & Computer Science (Mathematical Modelling and AI)  
**Application:** SMILES 2026 — Summer School of Machine Learning, Skoltech

---

## Overview

This project builds a multi-class text classifier that automatically assigns news headlines to one of five categories: **Technology, Science, Politics, Sports, Economy**. The goal is to explore how classical NLP techniques — TF-IDF vectorization and logistic regression — perform on short, information-dense texts, and to understand where they succeed and where they fail.

Short texts like headlines are a particularly challenging NLP problem: there are very few words per document, domain vocabulary overlaps (e.g., "energy" appears in both Technology and Economy), and context is nearly absent. This makes the task a good testbed for understanding the limits of bag-of-words approaches.

---

## Dataset

| Property | Value |
|---|---|
| Total headlines | 150 |
| Categories | 5 (Technology, Science, Politics, Sports, Economy) |
| Headlines per category | 30 |
| Source | Manually curated, realistic news-style headlines |

The dataset was created manually to reflect real-world distribution of news content and vocabulary overlap between domains.

---

## Method

### Pipeline

```
Raw headline → Text cleaning → TF-IDF vectorization → Logistic Regression → Category
```

**Step 1 — Text Cleaning**  
Lowercase conversion, removal of punctuation and digits. Short headlines require minimal preprocessing to preserve signal.

**Step 2 — TF-IDF Vectorization**  
TF-IDF (Term Frequency–Inverse Document Frequency) converts each headline into a numerical vector. Words that appear frequently in one category but rarely across others receive higher weights. Bigrams (2-word combinations) are included to capture phrases like "interest rates" or "world record."

$$\text{TF-IDF}(t, d) = \text{tf}(t,d) \cdot \log\frac{N}{df(t)}$$

Where $N$ is the total number of documents and $df(t)$ is the number of documents containing term $t$.

**Step 3 — Logistic Regression**  
A one-vs-rest logistic regression classifier. For each category $k$, it learns a weight vector $\mathbf{w}_k$ such that:

$$P(y = k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T \mathbf{x}}}{\sum_j e^{\mathbf{w}_j^T \mathbf{x}}}$$

Logistic regression is interpretable: the learned weights directly tell us which words are most predictive of each category.

---

## Results

### 1. Dataset & Cross-Validation

![Overview](results/fig1_overview.png)

The dataset is balanced: 30 headlines per category. Five-fold cross-validation yields **mean accuracy of 64.7% ± 6.9%**. For a dataset of 150 short texts with 5 overlapping categories, this is a meaningful result — a random baseline would achieve 20%.

### 2. Confusion Matrix

![Confusion Matrix](results/fig2_confusion.png)

The model performs best on **Sports** (distinctive vocabulary: "championship", "stadium", "tournament") and **Economy** (specific terms: "inflation", "mortgage", "unemployment"). The main confusion is between **Technology** and **Science** — both categories discuss research, discoveries, and innovation, making them hard to separate from word patterns alone.

### 3. Most Predictive Features

![Top Features](results/fig3_features.png)

The feature importance plot reveals the model's "vocabulary" for each class. Sports: "world record", "championship". Economy: "central bank", "inflation". Technology: "AI", "neural". Science: "researchers", "scientists". Politics: "parliament", "government". This matches human intuition — a sign the model is learning meaningful patterns.

---

## Error Analysis

| Confusion pair | Reason |
|---|---|
| Technology ↔ Science | Shared vocabulary: "researchers", "AI", "discover", "develop" |
| Economy ↔ Politics | Shared context: government policies, budget debates |
| Sports (correct) | Domain-specific vocabulary is highly distinctive |

The Technology/Science boundary is the hardest — both domains share the language of discovery and innovation. This is where context matters more than individual words, which is exactly what transformer-based models (BERT, RoBERTa) are designed to capture.

---

## Limitations & Future Work

1. **Dataset size:** 150 headlines is small. With 10,000+ examples, accuracy would likely exceed 90%.
2. **Bag-of-words assumption:** TF-IDF treats each word independently, losing word order and context. The phrase "not bad" is treated the same as "bad".
3. **Boundary categories:** Technology and Science share too much vocabulary for a word-frequency approach to reliably separate.
4. **Next steps:**
   - Fine-tune a pre-trained transformer (e.g., `distilbert-base-uncased`) on a larger headline dataset
   - Apply topic modelling (LDA, BERTopic) to discover latent themes beyond predefined categories
   - Build a simple web interface for real-time headline classification

---

## How to Run

```bash
# 1. Clone the repository
git clone <repo-url>
cd news-headline-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run analysis (generates all figures and results/)
python analysis.py
```

---

## Repository Structure

```
.
├── analysis.py              # Full pipeline: data → model → evaluation → plots
├── headlines.csv            # Dataset (150 headlines, 5 categories)
├── requirements.txt         # Python dependencies
├── SOLUTION.md              # This report
└── results/
    ├── fig1_overview.png    # Class distribution & CV scores
    ├── fig2_confusion.png   # Confusion matrix
    ├── fig3_features.png    # Top TF-IDF features per category
    └── summary.json         # Key metrics
```

---

## Key Metrics

| Metric | Value |
|---|---|
| Test Accuracy | 53.3% |
| 5-Fold CV Accuracy | 64.7% ± 6.9% |
| Best category (F1) | Sports — 0.67 |
| Hardest category (F1) | Technology — 0.31 |
| Random baseline | 20.0% |

---

## Dependencies

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `scikit-learn` | TF-IDF, Logistic Regression, evaluation |
| `matplotlib`, `seaborn` | Visualization |

---

*This project was developed as part of the SMILES 2026 application process.*
