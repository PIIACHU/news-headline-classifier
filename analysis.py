"""
News Headline Classifier
========================
Multi-class text classification of news headlines using TF-IDF + Logistic Regression.
Author: Daria Chernomazova | SMILES 2026 Application Project
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
import re

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════════════════
headlines = {
    "Technology": [
        "OpenAI releases new language model with improved reasoning",
        "Apple unveils updated MacBook Pro with faster chip",
        "Google announces major changes to its search algorithm",
        "Researchers develop new battery technology for electric vehicles",
        "Microsoft integrates AI assistant into Office suite",
        "Meta launches virtual reality headset for consumers",
        "Tesla self-driving software update causes controversy",
        "Amazon expands cloud computing services globally",
        "New programming language gains popularity among developers",
        "Cybersecurity breach affects millions of user accounts",
        "Quantum computing milestone achieved by IBM researchers",
        "Startup raises funds to build autonomous delivery robots",
        "Social media platform introduces new content moderation tools",
        "Scientists create artificial neurons that mimic the brain",
        "5G network expansion accelerates across major cities",
        "Neural network achieves superhuman performance on reasoning task",
        "Open source model outperforms proprietary alternatives",
        "Tech giant faces antitrust investigation in Europe",
        "Wearable device monitors health metrics in real time",
        "AI tool generates code faster than human developers",
    ],
    "Science": [
        "Scientists discover new species of deep-sea creature",
        "Mars rover finds evidence of ancient water on the planet",
        "New study links sleep deprivation to memory loss",
        "Astronomers detect gravitational waves from neutron star merger",
        "Researchers develop vaccine effective against multiple flu strains",
        "Climate study reveals accelerating Arctic ice melt",
        "New fossil discovery reshapes understanding of human evolution",
        "Gene editing technique shows promise for treating rare diseases",
        "Scientists map complete neural connectome of a mouse brain",
        "Telescope captures clearest image of a distant galaxy cluster",
        "Research confirms plastic pollution in deep ocean trenches",
        "New antibiotic discovered that targets drug-resistant bacteria",
        "Study finds link between gut microbiome and mental health",
        "Physicists observe rare quantum phenomenon in laboratory",
        "Ocean temperatures hit record high for the third consecutive year",
        "New exoplanet found in habitable zone of nearby star",
        "Breakthrough in nuclear fusion brings clean energy closer",
        "Scientists synthesize new material stronger than diamond",
        "Research reveals how migratory birds navigate using magnetic fields",
        "Human genome project expansion reveals millions of new variants",
    ],
    "Politics": [
        "Parliament votes on controversial new legislation",
        "President signs executive order on immigration policy",
        "Election results overturned after recount in key district",
        "International summit addresses global security concerns",
        "Opposition party demands transparency in government spending",
        "Senate debates new healthcare reform bill",
        "Diplomatic tensions rise between two neighboring countries",
        "Prime minister announces early general election",
        "Trade agreement signed between major economic powers",
        "Protests erupt over proposed changes to pension system",
        "Government introduces emergency measures amid economic downturn",
        "Foreign minister meets counterparts to discuss ceasefire",
        "Supreme court rules on landmark civil rights case",
        "Mayor faces recall vote after corruption scandal",
        "New sanctions imposed on authoritarian regime",
        "Coalition government collapses amid internal disputes",
        "Whistleblower reveals classified intelligence documents",
        "Regional autonomy movement gains momentum in disputed territory",
        "Officials warn of foreign interference in upcoming election",
        "United Nations Security Council holds emergency session",
    ],
    "Sports": [
        "Local team wins national championship after dramatic final",
        "Star athlete breaks world record at international competition",
        "Coach resigns following team's worst season in a decade",
        "Transfer deal sends top footballer to rival club",
        "Olympic committee announces host city for upcoming games",
        "Tennis player advances to grand slam final after five-set match",
        "Basketball team secures playoff spot with last-minute victory",
        "Doping scandal rocks professional cycling circuit",
        "Record crowd attends opening match of new stadium",
        "Underdog team defeats defending champions in surprise upset",
        "Marathon runner collapses but finishes race to standing ovation",
        "New rules introduced to improve safety in contact sports",
        "Esports tournament draws millions of online viewers worldwide",
        "Retired legend inducted into sport's hall of fame",
        "Rugby world cup final ends in rare draw after extra time",
        "Young sprinter sets national under-20 record at regional meet",
        "Club announces record revenue despite declining ticket sales",
        "Referee controversy dominates post-match analysis",
        "Swimmer qualifies for Olympics after injury comeback",
        "Women's league attendance surpasses men's for first time",
    ],
    "Economy": [
        "Central bank raises interest rates to combat inflation",
        "Unemployment rate falls to historic low amid job market boom",
        "Stock markets tumble on fears of global recession",
        "New budget proposal cuts social spending by fifteen percent",
        "Startup ecosystem thrives as venture capital investment rises",
        "Oil prices surge following production cuts by major exporters",
        "Retail sector struggles as consumer confidence declines",
        "Government announces stimulus package to boost growth",
        "Housing market cools as mortgage rates hit decade high",
        "Trade deficit widens amid supply chain disruptions",
        "Major corporation announces thousands of layoffs amid restructuring",
        "Currency devaluation sparks economic uncertainty in emerging market",
        "New tax reform simplifies brackets for middle-income earners",
        "Inflation eases for second consecutive month according to data",
        "Foreign direct investment reaches record high in manufacturing sector",
        "Gig economy workers push for better protections and benefits",
        "Energy prices fall as renewable capacity expands globally",
        "Bond yields rise as investors anticipate tighter monetary policy",
        "Small businesses report difficulty accessing credit from banks",
        "Global supply chains show resilience after pandemic disruption",
    ],
}

rows = []
for category, items in headlines.items():
    for h in items:
        rows.append({"headline": h, "category": category})

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("headlines.csv", index=False)
print(f"Dataset: {len(df)} headlines · {df['category'].nunique()} categories")
print(df["category"].value_counts().to_dict(), "\n")

# ══════════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════
def clean(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

df["clean"] = df["headline"].apply(clean)

# ══════════════════════════════════════════════════════════════════════
# 3. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["category"], test_size=0.2, random_state=42, stratify=df["category"])
print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# ══════════════════════════════════════════════════════════════════════
# 4. PIPELINE: TF-IDF + LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════════
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000,
                              sublinear_tf=True, stop_words="english")),
    ("clf",   LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cv  = cross_val_score(pipeline, df["clean"], df["category"], cv=5, scoring="accuracy")
print(f"Test Accuracy:  {acc:.3f}")
print(f"5-Fold CV:      {cv.mean():.3f} ± {cv.std():.3f}\n")
print(classification_report(y_test, y_pred))

# ══════════════════════════════════════════════════════════════════════
# 5. FIGURE 1 — Class distribution
# ══════════════════════════════════════════════════════════════════════
ACCENT = "#7B4FD4"
PALETTE = ["#7B4FD4", "#4A90D9", "#50C878", "#F5A623", "#E85D5D"]
sns.set_theme(style="whitegrid"); plt.rcParams["figure.facecolor"] = "white"

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("News Headline Classifier — Dataset & Predictions",
             fontsize=13, fontweight="bold", color="#1A1A2E", y=1.01)

counts = df["category"].value_counts()
axes[0].bar(counts.index, counts.values, color=PALETTE, edgecolor="white", width=0.5)
axes[0].set_title("Class Distribution", fontsize=11)
axes[0].set_ylabel("Number of Headlines")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 0.3, str(v), ha="center", fontsize=10, fontweight="bold")

# prediction confidence on test set
proba = pipeline.predict_proba(X_test)
max_conf = proba.max(axis=1)
axes[1].hist(max_conf, bins=15, color=ACCENT, edgecolor="white", alpha=0.85)
axes[1].set_title("Prediction Confidence Distribution", fontsize=11)
axes[1].set_xlabel("Max Class Probability"); axes[1].set_ylabel("Count")
axes[1].axvline(max_conf.mean(), color="#E85D5D", linewidth=1.5,
                linestyle="--", label=f"Mean: {max_conf.mean():.2f}")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/fig1_distribution_confidence.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig1")

# ══════════════════════════════════════════════════════════════════════
# 6. FIGURE 2 — Confusion matrix
# ══════════════════════════════════════════════════════════════════════
cats = sorted(df["category"].unique())
cm = confusion_matrix(y_test, y_pred, labels=cats)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=cats, yticklabels=cats,
            linewidths=0.5, linecolor="#ddd", ax=ax,
            annot_kws={"fontsize": 11})
ax.set_title("Confusion Matrix (Test Set)", fontsize=12,
             fontweight="bold", color="#1A1A2E")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.tight_layout()
plt.savefig("results/fig2_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig2")

# ══════════════════════════════════════════════════════════════════════
# 7. FIGURE 3 — Top TF-IDF features per class
# ══════════════════════════════════════════════════════════════════════
tfidf  = pipeline.named_steps["tfidf"]
clf    = pipeline.named_steps["clf"]
feats  = np.array(tfidf.get_feature_names_out())
TOP_N  = 8

fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)
fig.suptitle("Top TF-IDF Features per Category",
             fontsize=12, fontweight="bold", color="#1A1A2E")

for ax, (cls, color) in zip(axes, zip(clf.classes_, PALETTE)):
    idx  = list(clf.classes_).index(cls)
    coef = clf.coef_[idx]
    top  = np.argsort(coef)[-TOP_N:][::-1]
    ax.barh(feats[top][::-1], coef[top][::-1], color=color, edgecolor="white")
    ax.set_title(cls, fontsize=10, fontweight="bold")
    ax.set_xlabel("Coefficient")
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("results/fig3_top_features.png", dpi=150, bbox_inches="tight")
plt.close(); print("Saved fig3")

# ══════════════════════════════════════════════════════════════════════
# 8. DEMO PREDICTIONS
# ══════════════════════════════════════════════════════════════════════
demo = [
    "Scientists discover new planet in habitable zone",
    "Stock market drops amid inflation fears",
    "New AI model beats human experts at medical diagnosis",
    "Athletes break records at world championship",
    "Parliament debates controversial budget proposal",
]

print("\n── Demo Predictions ──")
for h in demo:
    pred  = pipeline.predict([clean(h)])[0]
    conf  = pipeline.predict_proba([clean(h)]).max()
    print(f"  [{pred:12s} {conf:.0%}] {h}")

# ══════════════════════════════════════════════════════════════════════
# 9. SAVE SUMMARY
# ══════════════════════════════════════════════════════════════════════
summary = {
    "total_headlines":  int(len(df)),
    "categories":       int(df["category"].nunique()),
    "test_accuracy":    round(float(acc), 4),
    "cv_mean":          round(float(cv.mean()), 4),
    "cv_std":           round(float(cv.std()), 4),
    "mean_confidence":  round(float(max_conf.mean()), 4),
}
with open("results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\n✅ All outputs saved to results/")
print(json.dumps(summary, indent=2))
