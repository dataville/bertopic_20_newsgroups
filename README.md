# BERTopic · BERTrend Tutorial

A self-directed, step-by-step tutorial covering topic modelling with [BERTopic](https://maartengoovaerts.github.io/BERTopic/)

---

## Repository Contents

| File | Description |
|------|-------------|
| `newsgroups.ipynb` | BERTopic pipeline applied to the 20 Newsgroups dataset |
| `bertopic_20newsgroups/` | Saved BERTopic model (safetensors format) |
| `bertopic_env.yaml` | Conda environment specification |

---

## Notebook 1 — BERTopic on 20 Newsgroups

**Dataset:** [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) — 18,846 Usenet posts across 20 categories. Headers, footers, and quoted replies stripped. 515 documents removed after stripping left them empty, leaving 18,331 for modelling.

### Pipeline

| Step | Tool | Detail |
|------|------|--------|
| Embeddings | `all-MiniLM-L6-v2` | 384-dimensional sentence embeddings |
| Dimensionality reduction | UMAP | 384-dim → 5-dim for clustering; separate 2-dim projection for visualisation |
| Clustering | HDBSCAN | 154 topics discovered; 7,170 outliers (39.1%) before reduction |
| Keyword extraction | c-TF-IDF | Three representations compared: baseline, stop word removal, spaCy lemmatisation |
| Outlier reduction | Embedding similarity | All 7,170 outliers reassigned; 0 remaining |
| Serialisation | safetensors | Model saved to `bertopic_20newsgroups/` |

### Sections

1. Environment and imports
2. Dataset loading
3. Empty document filtering and embedding model setup
4. Encoding the full corpus
5. Dimensionality reduction with UMAP
6. Clustering with HDBSCAN
7. Fitting BERTopic with pre-computed artefacts
8. Inspecting topics
9. Reducing outliers
10. Topic representation — baseline vs. stop word removal vs. lemmatisation
11. Saving the model

---

## Dependencies

| Package | Source | Role |
|---------|--------|------|
| `bertopic` | pip | Topic modelling framework |
| `sentence-transformers` | pip | Document embeddings |
| `umap-learn` | conda-forge | Dimensionality reduction |
| `hdbscan` | conda-forge | Density-based clustering |
| `spacy` | conda-forge | Lemmatisation |
| `pytorch` | conda-forge | Tensor operations, MPS/CUDA device support |
| `scikit-learn` | conda-forge | CountVectorizer, dataset loader |
| `bertrend` | pip | Temporal topic tracking (next notebook) |
| `datasets` | pip | HuggingFace dataset access (`cc_news`) |
| `safetensors` | pip | Model serialisation |
