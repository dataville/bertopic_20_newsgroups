# BERTopic — 20 Newsgroups Tutorial

A step-by-step walkthrough of the [BERTopic](https://maartengoovaerts.github.io/BERTopic/) pipeline applied to the classic **20 Newsgroups** text dataset. The notebook deliberately separates each stage of the pipeline so the output at every step can be inspected before invoking the full model.

---

## Repository Contents

| File | Description |
| ---- | ----------- |
| `newsgroups.ipynb` | BERTopic pipeline applied to the 20 Newsgroups dataset |
| `bertopic_20newsgroups/` | Saved BERTopic model (safetensors format) |
| `bertopic_env.yaml` | Conda environment specification |

---

## Notebook — BERTopic on 20 Newsgroups

**Dataset:** [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/) — 18,846 Usenet posts across 20 categories. Headers, footers, and quoted replies stripped. 515 documents removed after stripping left them empty, leaving 18,331 for modelling.

### Pipeline

| Step | Tool | Detail |
| ---- | ---- | ------ |
| Embeddings | `all-MiniLM-L6-v2` | 384-dimensional sentence embeddings; encoded once and passed explicitly to BERTopic to avoid re-encoding |
| Dimensionality reduction | UMAP | 384-dim → 5-dim for clustering (`cosine` metric, `min_dist=0.0`); separate 2-dim projection for visualisation only |
| Clustering | HDBSCAN | 154 topics discovered; 7,170 outliers (39.1%) before reduction |
| Keyword extraction | c-TF-IDF | Three representations compared: baseline, stop word removal, spaCy lemmatisation |
| Outlier reduction | Embedding similarity | All 7,170 outliers reassigned; 0 remaining |
| Serialisation | safetensors | Model saved to `bertopic_20newsgroups/`; embedding model stored as a HuggingFace reference, not full weights |

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
| ------- | ------ | ---- |
| `bertopic` | pip | Topic modelling framework |
| `sentence-transformers` | pip | Document embeddings |
| `umap-learn` | conda-forge | Dimensionality reduction |
| `hdbscan` | conda-forge | Density-based clustering |
| `spacy` | conda-forge | Lemmatisation (`en_core_web_sm`) |
| `pytorch` | conda-forge | Tensor operations; MPS/CUDA/CPU device support |
| `scikit-learn` | conda-forge | CountVectorizer, dataset loader |
| `safetensors` | pip | Model serialisation |
