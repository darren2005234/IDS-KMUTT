# Literature Review Notes — IDS-KMUTT Hybrid IDS Project

**Author:** Darren Touopi  
**Programme:** BAC+4 — Sécurité et Qualité des Réseaux (SQR), Polytech Dijon  
**Last updated:** 18/05/2026  

---

## Reading List

| # | Paper | Authors | Year | Status |
|---|---|---|---|---|
| 1 | Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization | Sharafaldin, Lashkari, Ghorbani | 2018 | ✅ Read |
| 2 | A Survey of Network-based Intrusion Detection Data Sets | Ring, Wunderlich, Scheuring, Landes, Hotho | 2019 | ✅ Read |
| 3 | Deep Learning for Cyber Security Intrusion Detection | Ferrag et al. | 2020 | ✅ Read |
| 4 | Random Forests | Breiman | 2001 | ✅ Read |
| 5 | XGBoost: A Scalable Tree Boosting System | Chen & Guestrin | 2016 | ✅ Read |
| 6 | Long Short-Term Memory | Hochreiter & Schmidhuber | 1997 | ✅ Read |

---

## Paper 1 — Sharafaldin et al. (2018)

**Full title:** Toward Generating a New Intrusion Detection Dataset
and Intrusion Traffic Characterization  
**Authors:** Iman Sharafaldin, Arash Habibi Lashkari, Ali A. Ghorbani  
**Institution:** Canadian Institute for Cybersecurity (CIC),
University of New Brunswick (UNB), Canada  
**Conference:** 4th International Conference on Information Systems
Security and Privacy (ICISSP 2018), pages 108–116  
**DOI:** 10.5220/0006639801080116  
**Read on:** 28/04/2026  

### Summary
This paper introduces the CICIDS2017 dataset — the primary dataset
used in this project. The authors argue that existing IDS datasets
(DARPA, KDD99, ISCX2012, etc.) are outdated, heavily anonymized,
and lack traffic diversity, making them unsuitable for evaluating
modern IDS systems. They propose a new evaluation framework based
on 11 criteria and generate a dataset that satisfies all of them.

### Methodology
The authors designed two isolated networks — a Victim-Network and
an Attack-Network. The Victim-Network included a complete
infrastructure (router, firewall, switches, Windows/Linux/Mac
machines). The Attack-Network used Kali Linux machines to execute
attacks. A B-Profile agent generated realistic benign traffic based
on the behaviour of 25 users across HTTP, HTTPS, FTP, SSH and email
protocols. Traffic was captured over 5 days (Monday July 3 to
Friday July 7, 2017) using a mirror port on the main switch.
CICFlowMeter was used to extract 80 flow-level features from the
raw PCAP files.

### Dataset structure
| Day | Attack types |
|---|---|
| Monday | BENIGN only |
| Tuesday | FTP-Patator, SSH-Patator |
| Wednesday | DoS (Slowloris, Slowhttptest, Hulk, GoldenEye), Heartbleed |
| Thursday | Web Attacks (Brute Force, XSS, SQL Injection), Infiltration |
| Friday | Botnet (Ares), DDoS (LOIC), PortScan (multiple Nmap switches) |

### Attack types included
Brute Force (FTP, SSH), Heartbleed, Botnet, DoS (4 variants),
DDoS, Web Attacks (XSS, SQL Injection, Brute Force over HTTP),
Infiltration (via Metasploit + Nmap), PortScan.

### ML results from the paper
The authors evaluated 7 ML algorithms on the dataset.
Random Forest achieved the best balance of accuracy and speed:
F1=0.97, execution time=74.39 seconds. KNN achieved F1=0.96
but required 1908 seconds. ID3 achieved F1=0.98.
Naive-Bayes performed poorly (F1=0.04) due to feature
independence assumptions not holding for network traffic.

### Key findings for my project
- CICIDS2017 is the only dataset satisfying all 11 evaluation
  criteria — justified choice as primary dataset
- Random Forest is confirmed as a strong baseline for IDS —
  supports my choice of RF as first model in notebook 03
- The 80 features extracted by CICFlowMeter are well documented
  — my feature selection reduced these to 50 after variance
  and correlation filtering
- Rare classes (Heartbleed=11 samples, SQL Injection=21,
  Infiltration=36) are a known limitation of the dataset —
  excluded from multiclass benchmark, handled at binary level

### Important features per attack (from Table 3)
- DoS attacks: Flow IAT Min, Flow IAT Mean, Flow Duration
- Heartbleed: B.Packet Len Std, Subflow F.Bytes, Flow Duration
- Web Attacks: Init Win F.Bytes, Subflow F.Bytes
- DDoS: B.Packet Len Std, Avg Packet Size, Flow Duration
- PortScan: Init Win F.Bytes, B.Packets/s
These features align with those retained after my feature
selection in notebook 02.

### How I will cite this paper
Every time CICIDS2017 is mentioned in the report.
Every time CICFlowMeter is mentioned.
When justifying the choice of Random Forest as baseline model.

### Limitations noted by authors
- Dataset will need periodic updates as attacks evolve
- Some attack classes have very few samples
- Future work: increase number of PCs and add more recent attacks

---

## Paper 2 — Ring et al. (2019)

**Full title:** A Survey of Network-based Intrusion Detection Data Sets  
**Authors:** Markus Ring, Sarah Wunderlich, Deniz Scheuring,
Dieter Landes, Andreas Hotho  
**Institution:** Coburg University of Applied Sciences (Germany)
+ University of Würzburg (Germany)  
**Journal:** arXiv:1903.02460v2 — Computers & Security (Elsevier)  
**Published:** 2019  
**Read on:** 30/04/2026  

### Summary
This paper provides a comprehensive neutral survey of 34
network-based IDS datasets. Unlike Sharafaldin et al. (2018)
which promotes its own dataset, Ring et al. focus on
establishing a structured comparison framework. They define
15 properties grouped into 5 categories to assess dataset
suitability for specific evaluation scenarios. The paper
also covers data repositories and traffic generators as
alternative sources of network traffic.

### The 5 property categories and 15 properties
**General Information:** Year of creation, public availability,
normal traffic presence, attack traffic presence.  
**Nature of Data:** Metadata, format (packet/flow/other),
anonymity level.  
**Data Volume:** Count (number of flows/packets or GB), duration.  
**Recording Environment:** Kind of traffic (real/emulated/synthetic),
type of network, completeness of network.  
**Evaluation:** Predefined splits, balanced classes, labeling method.

### Key findings for my project

**On CICIDS2017:**
Ring et al. confirm CICIDS2017 as one of the most complete
and up-to-date datasets, satisfying all 15 properties including
public availability, metadata, no anonymization, real attack
diversity, and correct labeling. It is listed among 4 recommended
datasets for general evaluation alongside UNSW-NB15, CIDDS-001
and UGR16.

**On class imbalance:**
The paper explicitly states that real-world network traffic is
not balanced and contains more normal traffic than attacks.
This directly justifies my use of SMOTE oversampling to
balance CICIDS2017 before training.

**On dataset age:**
The paper notes that network traffic is subject to concept drift
and new attacks appear daily — benchmark datasets need
periodic updates. This is why CICIDS2017 (2017) is preferred
over KDD99 (1998) or NSL-KDD (1998) for my project.

**On using multiple datasets:**
Ring et al. recommend evaluating IDS methods on more than
one dataset to avoid overfitting to a specific dataset.
This supports my decision to mention UNSW-NB15 as a
validation dataset for future work.

**On NSL-KDD:**
NSL-KDD is confirmed as still widely used but based on
1998 traffic — outdated for modern network conditions.
Used only as a reference comparison in my benchmark.

**On predefined splits:**
The paper recommends benchmark datasets include predefined
train/test splits. For CICIDS2017 no predefined splits exist
— I apply stratified 80/20 split in notebook 03 to ensure
reproducibility.

**On data formats:**
Flow-based data (like CICIDS2017 bidirectional flows) is
the standard for ML-based IDS. Packet-based data requires
more storage and processing. My pipeline uses flow-based
features extracted by CICFlowMeter — consistent with
the recommendation.

**On Snort labeling:**
Some datasets (e.g. IRSC) use Snort IDS for labeling,
which introduces errors since no IDS is perfect.
CICIDS2017 uses manual labeling per attack schedule —
higher label quality, confirmed by Ring et al.

### Datasets mentioned that are relevant to my project
| Dataset | Relevance |
|---|---|
| CICIDS2017 | Primary dataset — confirmed best choice |
| NSL-KDD | Classic reference — cited for comparison only |
| UNSW-NB15 | Recommended alternative — future validation |
| ISCX2012 | Predecessor to CICIDS2017 — mentioned in related work |
| CTU-13 | Botnet-focused — not used but cited in related work |

### Data repositories of interest
- **Kaggle** — CICIDS2017 already uploaded, usable directly
  in notebooks without local download
- **Malware Traffic Analysis** — PCAP files of real malware
  traffic — useful for Test 2 and Test 3 (hping3/nmap
  validation in notebook 07)
- **CAIDA** — Large-scale DDoS traces — possible future work

### How I will cite this paper
- When justifying CICIDS2017 as primary dataset (alongside
  Sharafaldin 2018)
- When explaining class imbalance and the need for SMOTE
- When discussing limitations of older datasets (KDD99, NSL-KDD)
- When recommending UNSW-NB15 as future validation dataset
  in the Future Work section of the report

### Key quote for report
Ring et al. establish that no perfect IDS dataset exists,
but CICIDS2017 satisfies the most comprehensive set of
quality criteria among publicly available datasets —
directly supporting the choice made in this project.

### Limitations noted by authors
- No single perfect dataset exists or will likely ever exist
- Many datasets are outdated or anonymized
- Closer collaboration and a shared platform for IDS datasets
  would benefit the research community
- Predefined train/test splits are lacking in most datasets

---

## Notes for report structure

### Related Work section outline (based on papers 1 and 2)
1. Overview of IDS categories — signature-based vs anomaly-based
   vs hybrid (my contribution)
2. Dataset landscape — limitations of DARPA/KDD99 (Ring 2019),
   why CICIDS2017 was chosen (Sharafaldin 2018, Ring 2019)
3. ML approaches for IDS — RF baseline confirmed
   (Sharafaldin 2018 Table 4)
4. Class imbalance in IDS datasets — justification for SMOTE
   (Ring 2019 Section IV-E)
5. Hybrid IDS motivation — gap in literature (papers 3-6 pending)

### Papers still needed
- Paper 3 (Ferrag 2020): will cover deep learning approaches
  and LSTM justification
- Papers 4-6 (Breiman, Chen, Hochreiter): will cover
  algorithmic foundations of each model

---

## Paper 3 — Ferrag et al. (2020)

**Full title:** Deep Learning for Cyber Security Intrusion Detection:
Approaches, Datasets, and Comparative Study
**Authors:** Mohamed Amine Ferrag, Leandros Maglaras,
Sotiris Moschoyiannis, Helge Janicke
**Journal:** Journal of Information Security and Applications,
Volume 50, 2020, 102419 — Elsevier
**DOI:** 10.1016/j.jisa.2019.102419
**Available online:** 24 December 2019
**Read on:** 12/05/2026

### Summary
This paper presents a comprehensive survey of deep learning
approaches for cyber security intrusion detection. It reviews
IDS systems based on deep learning, classifies 35 public
cyber datasets into 7 categories, analyzes 7 deep learning
models, and provides a comparative study on two real traffic
datasets (CSE-CIC-IDS2018 and Bot-IoT). It is the first paper
to thoroughly cover approaches, datasets, AND a comparative
study of deep learning for IDS simultaneously.

### The 7 deep learning approaches covered
The paper classifies deep learning models into two families:

**Deep discriminative models (supervised):**
- Deep Neural Networks (DNN)
- Recurrent Neural Networks (RNN) — includes LSTM
- Convolutional Neural Networks (CNN)

**Generative/unsupervised models:**
- Restricted Boltzmann Machine (RBM)
- Deep Belief Networks (DBN)
- Deep Boltzmann Machines (DBM)
- Deep Auto-Encoders (DA)

### Key experimental results (Table 8 & 9)
Experiments conducted on CSE-CIC-IDS2018 and Bot-IoT datasets
using Google Colab with GPU, TensorFlow, Python 3.

**Best discriminative model: CNN**
- Accuracy: 97.376% on CSE-CIC-IDS2018 (100 hidden nodes, LR=0.5)
- Accuracy: 98.371% on Bot-IoT (100 hidden nodes, LR=0.5)
- Best overall detection rate (DR Overall): 97.28%

**RNN performance on CSE-CIC-IDS2018:**
- Accuracy: 97.310% (100 hidden nodes, LR=0.5)
- Best detection rate for 7 specific attack types including
  DoS Hulk (94.912%), DoS GoldenEye (98.330%),
  Infiltration (97.874%)
- Training time: always between DNN (fastest) and CNN (slowest)

**Best unsupervised model: Deep Auto-Encoder (DA)**
- DR Overall: 98.18% on CSE-CIC-IDS2018
- Accuracy: 98.394% on Bot-IoT (100 hidden nodes, LR=0.5)

**Random Forest comparison (Figure 12):**
All deep learning models outperform Random Forest in terms
of overall detection rate — directly justifies using deep
learning (LSTM) alongside RF and XGBoost in my project.

### Key findings for my project

**Justification of LSTM (RNN):**
The paper confirms that RNN-based approaches (including LSTM)
achieve competitive performance for intrusion detection —
accuracy above 97% — and specifically excel at detecting
temporally complex attacks like DoS and Infiltration where
sequential patterns matter. This directly justifies the
inclusion of LSTM as the third model in my hybrid IDS.

**Justification of CNN as alternative:**
CNN achieved the highest accuracy (97.376%) among discriminative
models. If LSTM training proves too costly even on the RTX 4090,
CNN is a validated alternative — this is my fallback option
noted in the risk mitigation plan.

**Deep learning vs Random Forest:**
Figure 12 clearly shows all deep learning models outperform
RF, NB, SVM, and ANN in overall detection rate. This is a
key result to cite in my benchmark section when comparing
RF (baseline) vs LSTM (deep learning) results.

**On CICIDS2017 vs CSE-CIC-IDS2018:**
Ferrag et al. conducted experiments on CSE-CIC-IDS2018 and
not CICIDS2017 — because their focus was on newer datasets.
However, their RNN results on CSE-CIC-IDS2018 provide a
performance baseline that validates the LSTM approach.
My results on CICIDS2017 will be compared against their
CSE-CIC-IDS2018 results in the discussion section.

**On dataset variety:**
The paper classifies 35 datasets into 7 categories. Notably:
- CICIDS2017 is listed as a network traffic-based dataset
  (Table 3, cited 87 times as of 2019)
- CSE-CIC-IDS2018 listed with 0 citations at publication time
  — confirms CICIDS2017 is the more established benchmark

**Hyperparameters used (Table 6):**
| Parameter | Value used |
|---|---|
| Learning rate | 0.01 to 0.5 |
| Epochs | 100 |
| Hidden nodes | 15 to 100 |
| Batch size | 1000 |
| Activation | Sigmoid |
| Classification | Softmax |
These will guide my LSTM hyperparameter tuning in notebook 05.

### RNN/LSTM papers cited that are useful for my related work
- Kim et al. [31] — LSTM on KDD99, 98.8% detection rate
- Yin et al. [34] — RNN on NSL-KDD, accuracy > RF with
  learning rate=0.1 and 80 hidden nodes
- Jiang et al. [36] — LSTM multi-channel IDS, 99.23%
  detection rate, 9.86% FAR on NSL-KDD
- Ferrag et al. [37] — RNN on CICIDS2017 ← same dataset
  as mine, direct reference for my related work section

### How I will cite this paper
- When justifying LSTM as third model (Section 3.1 of report)
- When comparing deep learning vs traditional ML (benchmark)
- When justifying hyperparameter choices in notebook 05
- When discussing CNN as alternative/fallback to LSTM
- When comparing CICIDS2017 vs CSE-CIC-IDS2018 (dataset section)

### Important note for my report
Ferrag et al. [37] (self-citation in this paper) applied RNN
on CICIDS2017 — this is a direct precedent for my LSTM
approach on the same dataset. I must find and cite this
specific paper in my related work section:
"Ferrag MA, Maglaras L. Deepcoin: a novel deep learning and
blockchain-based energy exchange framework for smart grids.
IEEE Trans. Eng. Manage. 2019." — Note: the CICIDS2017 RNN
work appears to be embedded in this paper.

### Limitations noted by authors
- Experiments only on CSE-CIC-IDS2018 and Bot-IoT — not
  CICIDS2017 (opportunity for my project to fill this gap)
- No predefined train/test splits used — same issue as mine
- Hyperparameter search is limited (only 4 values of HN
  and 3 values of LR tested)
- No hybrid IDS proposed — purely single-model comparison
  (gap that my project fills with the Fusion Engine)

---

## Paper 4 — Breiman (2001)

**Full title:** Random Forests
**Author:** Leo Breiman
**Institution:** Statistics Department, University of California,
Berkeley, CA 94720
**Journal:** Machine Learning, Volume 45, pages 5–32, 2001
**Publisher:** Kluwer Academic Publishers
**Read on:** 14/05/2026

### Summary
This is the original paper introducing the Random Forest algorithm.
Breiman defines a Random Forest as a collection of tree classifiers
where each tree depends on a random vector sampled independently,
and each tree casts a unit vote for the most popular class. The paper
proves theoretically that Random Forests converge and do not overfit
as the number of trees increases, and demonstrates empirically that
they compare favorably to Adaboost while being more robust to noise
and significantly faster to train.

### Formal definition — to cite in report
A random forest is a classifier consisting of a collection of
tree-structured classifiers {h(x, Theta_k), k=1,...} where the
{Theta_k} are independent identically distributed random vectors
and each tree casts a unit vote for the most popular class at input x.

### Key theoretical findings

**1. Random Forests do not overfit**
By the Strong Law of Large Numbers, the generalization error PE*
converges almost surely to a limit as the number of trees increases.
This is a fundamental advantage over single decision trees.

**2. The two key parameters — strength and correlation**
The generalization error upper bound is:
PE* <= rho_bar * (1 - s^2) / s^2

Where:
- s = strength of individual trees (higher is better)
- rho_bar = mean correlation between trees (lower is better)
- c/s^2 ratio = rho_bar / s^2 → the smaller, the better

Random feature selection at each node reduces inter-tree
correlation without significantly reducing individual tree
strength — this is the core mechanism behind RF accuracy.

**3. Forest-RI — Random Input selection**
At each node, F features are randomly selected from all M features.
Recommended value: F = int(log2(M) + 1)

For CICIDS2017 with M=50 features after selection:
F = int(log2(50) + 1) = 6 features per split

**4. Out-of-bag (OOB) error estimation**
Each tree is trained on a bootstrap sample — approximately 1/3 of
the training data is left out (out-of-bag). OOB samples provide
a free internal estimate of generalization error without a
separate validation set, as accurate as a test set of equal size.

In scikit-learn: set oob_score=True to obtain this estimate.

### Performance results (Table 2)
Across 19 benchmark datasets, Random Forest achieves error rates
comparable to Adaboost and significantly better than single trees,
while being:
- Up to 40x faster than Adaboost (zip-code dataset)
- More robust to label noise (Table 4 — 5% noise increases
  RF error by only 1-8% vs up to 48.9% for Adaboost)
- Simpler to implement and naturally parallelizable

### Variable importance
RF computes variable importance by permuting each feature in OOB
samples and measuring the resulting increase in misclassification
rate. Features causing large error increases when permuted are
the most discriminative. This is directly applicable to
identifying the most important network traffic features in
CICIDS2017 for intrusion detection.

```python
# In notebook 03
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
# Plot top 20 most discriminative features
```

### Properties of Random Forests relevant to IDS

| Property | Relevance to IDS project |
|---|---|
| No overfitting | Critical for 4.5M row dataset |
| No feature scaling needed | Tree-based — no normalization required |
| Built-in feature importance | Identifies key network traffic features |
| Robust to noise | Important for real network traffic |
| Fast training | 74.39s on CICIDS2017 (Sharafaldin 2018) |
| Handles high dimensionality | Works well with 50 flow-level features |

### Recommended hyperparameters for notebook 03

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,        # Breiman uses 100 trees
    max_features='log2',     # F = log2(M)+1 — Breiman's recommendation
    oob_score=True,          # free internal validation estimate
    class_weight='balanced', # handles residual class imbalance
    n_jobs=-1,               # parallel training on all CPU cores
    random_state=42          # reproducibility
)
```

### How this paper will be cited in the report
- Introduction of Random Forest as first model (notebook 03)
- Justification of n_estimators=100 and max_features='log2'
- Explanation of OOB error as internal validation method
- Feature importance analysis of CICIDS2017 flow features
- Argument that RF does not overfit on large datasets
- Related Work section — algorithmic foundation of model 1

### Limitations noted by the author
- Random Forests are less interpretable than single decision trees
  — described as an "impenetrable black box" in section 10
- Variable importance can be misleading when correlated features
  are present — mitigated in this project by correlation filtering
  applied in notebook 02 (78 → 50 features)
- Regression forests require more features than classification
  forests — not applicable to the binary classification task

---

## Paper 5 — Chen & Guestrin (2016)

**Full title:** XGBoost: A Scalable Tree Boosting System
**Authors:** Tianqi Chen, Carlos Guestrin
**Institution:** University of Washington
**Conference:** KDD '16 — August 13-17, 2016, San Francisco, CA, USA
**arXiv:** 1603.02754v3
**Read on:** 15/05/2026

### Summary
This paper introduces XGBoost (eXtreme Gradient Boosting), a
scalable end-to-end tree boosting system that has become the
de-facto standard for structured data machine learning
competitions and production pipelines. Among 29 Kaggle winning
solutions in 2015, 17 used XGBoost. The system combines a
regularized learning objective, a novel sparsity-aware split
finding algorithm, weighted quantile sketch for approximate
tree learning, and cache-aware block structures to achieve
performance more than 10x faster than scikit-learn's GBM
implementation while maintaining state-of-the-art accuracy.

### How XGBoost differs from standard Gradient Boosting

Standard Gradient Boosting (GBM) minimizes a loss function
by adding trees sequentially. XGBoost introduces three key
improvements:

**1. Regularized objective function**
L(phi) = sum_i l(y_hat_i, y_i) + sum_k Omega(f_k)
where Omega(f) = gamma*T + (1/2)*lambda*||w||^2

The regularization term Omega penalizes model complexity
by penalizing the number of leaves T and the magnitude of
leaf weights w. This prevents overfitting and produces
simpler, more generalizable trees. When lambda=0 and
gamma=0, XGBoost reduces to standard gradient boosting.

**2. Second-order gradient approximation**
XGBoost uses both first-order (g_i) and second-order (h_i)
gradient statistics to optimize the objective at each step.
This provides a better approximation than standard GBM
which uses only first-order gradients, enabling faster
convergence with fewer trees.

The optimal leaf weight for leaf j is:
w*_j = -( sum_{i in I_j} g_i ) / ( sum_{i in I_j} h_i + lambda )

The quality score of a tree structure is:
L_tilde(q) = -(1/2) * sum_j [ (sum g_i)^2 / (sum h_i + lambda) ] + gamma*T

**3. Shrinkage and column subsampling**
- Shrinkage (learning rate eta): scales newly added weights
  after each boosting step — reduces influence of each tree
  and leaves room for future trees to improve
- Column subsampling: randomly selects a subset of features
  at each split — borrowed from Random Forest, prevents
  overfitting even more than row subsampling

### Key algorithmic innovations

**Exact Greedy Algorithm**
Enumerates all possible splits on all features. Requires
sorting data by feature value. Used for single-machine
settings. XGBoost runs this 10x faster than scikit-learn
(0.68s vs 28.51s per tree on Higgs-1M, Table 3).

**Approximate Algorithm with Weighted Quantile Sketch**
Proposes candidate split points using percentiles of feature
distribution. Uses a novel weighted quantile sketch that
handles instance weights with provable theoretical guarantee
— the first such algorithm. Two variants:
- Global: proposes candidates once at tree construction
- Local: re-proposes candidates after each split (fewer
  candidates needed, better for deep trees)

**Sparsity-aware Split Finding**
Handles missing values, zero entries, and one-hot encoded
features in a unified way. Each tree node learns a default
direction for missing values. Complexity is linear in the
number of non-missing entries. 50x faster than naive
implementation on sparse data (Figure 5).

Directly relevant to CICIDS2017: after feature selection,
some flows may have zero values for certain features
(e.g., flags not present in UDP flows). XGBoost handles
this natively without imputation.

**Column Block for Parallel Learning**
Data stored in compressed column (CSC) format, sorted
once before training and reused across all iterations.
Enables parallel split finding across columns. Supports
column subsampling efficiently.

**Cache-aware Access**
Prefetching algorithm allocates internal buffer per thread
to fetch gradient statistics, reducing cache miss overhead.
2x speedup on large datasets (10M+ examples).

### Performance results

**vs scikit-learn GBM (Table 3 — Higgs 1M, 500 trees):**
| Method | Time per tree | Test AUC |
|---|---|---|
| XGBoost | 0.68s | 0.8304 |
| scikit-learn | 28.51s | 0.8302 |
| R GBM | 1.03s | 0.6224 |

XGBoost achieves same accuracy as scikit-learn at 42x speed.

**Scalability:**
- Single machine: handles 10M+ examples
- Out-of-core: processes 1.7 billion examples on one machine
- Distributed: scales linearly with number of machines
  (Figure 13 — 4 to 32 machines on 1.7B criteo dataset)

### Comparison with Random Forest

| Aspect | Random Forest (Breiman 2001) | XGBoost (Chen 2016) |
|---|---|---|
| Training strategy | Parallel independent trees | Sequential boosting |
| Bias-variance | Reduces variance | Reduces both bias and variance |
| Overfitting | Robust by design | Controlled by regularization |
| Missing values | Requires imputation | Native handling |
| Speed | Fast | 10x+ faster than GBM |
| Hyperparameters | Few (n_estimators, max_features) | More (eta, max_depth, lambda, gamma...) |
| Interpretability | Feature importance | Feature importance + SHAP |

### Recommended hyperparameters for notebook 04

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=500,        # number of trees — more than RF
    max_depth=8,             # Chen et al. use max_depth=8
    learning_rate=0.1,       # eta — shrinkage factor
    subsample=0.8,           # row subsampling
    colsample_bytree=0.8,    # column subsampling per tree
    reg_lambda=1.0,          # L2 regularization (lambda)
    reg_alpha=0.0,           # L1 regularization (alpha)
    scale_pos_weight=1,      # for imbalanced classes if needed
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1,
    tree_method='hist'       # approximate algorithm — faster
)
```

For hyperparameter tuning with Optuna in notebook 04:
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
    }
    # cross-validation and return F1 score
```

### How this paper will be cited in the report
- Introduction of XGBoost as second model (notebook 04)
- Justification of regularized objective vs standard GBM
- Justification of max_depth=8 and learning_rate=0.1
- When explaining sparsity handling for CICIDS2017 features
- Comparison table RF vs XGBoost vs LSTM in benchmark section
- When arguing XGBoost reduces both bias and variance
  (advantage over RF which only reduces variance)

### Key result for the benchmark section
Table 3 shows XGBoost achieves AUC=0.8304 vs scikit-learn
GBM AUC=0.8302 at 42x the speed. This demonstrates that
XGBoost provides state-of-the-art accuracy with practical
training times — directly justifying its inclusion in the
hybrid IDS benchmark alongside Random Forest and LSTM.

### Limitations noted by the authors
- Exact greedy algorithm does not scale to datasets that
  exceed memory — approximate algorithm needed for very
  large datasets (relevant for the 4.5M row CICIDS2017
  post-SMOTE dataset → use tree_method='hist')
- More hyperparameters than Random Forest — requires more
  careful tuning (addressed by Optuna in notebook 04)
- Less interpretable than a single decision tree

---

## Paper 6 — Hochreiter & Schmidhuber (1997)

**Full title:** Long Short-Term Memory
**Authors:** Sepp Hochreiter, Jürgen Schmidhuber
**Institution:** Fakultät für Informatik, Technische Universität
München, Germany + IDSIA, Lugano, Switzerland
**Journal:** Neural Computation, Volume 9, Issue 8,
pages 1735–1780, November 1997
**Publisher:** MIT Press
**DOI:** 10.1162/neco.1997.9.8.1735
**Read on:** 18/05/2026

### Summary
This is the foundational paper that introduces the Long
Short-Term Memory (LSTM) architecture. The authors identify
and solve the fundamental problem of standard recurrent
neural networks (RNNs): the vanishing and exploding gradient
problem that prevents learning long-term temporal dependencies.
LSTM addresses this through a novel architecture using memory
cells with multiplicative gating units, allowing constant
error flow through the network over thousands of time steps.
This paper has become one of the most cited works in deep
learning and is the foundation of nearly all modern sequence
modeling applications.

### The fundamental problem solved — vanishing gradient

In standard RNNs, gradients propagate through time via
repeated multiplication by the weight matrix W. When the
spectral radius of W is less than 1, gradients shrink
exponentially with the number of time steps — making it
impossible to learn dependencies spanning more than ~10
time steps. When the spectral radius is greater than 1,
gradients explode.

Hochreiter & Schmidhuber prove that with standard RNNs:
- Error signals "flowing backwards in time" tend to either
  blow up or vanish
- The temporal evolution of the backpropagated error
  exponentially depends on the size of the weights
- Bridging long time lags requires constant error flow

### The LSTM solution — Constant Error Carousel (CEC)

The core innovation is the Constant Error Carousel — a
self-connected linear unit with a fixed weight of 1.0.
This ensures that error signals propagate backwards in
time without decay or explosion across arbitrary time lags.

To prevent the CEC from being corrupted by irrelevant
inputs and to control when its content should affect the
output, the authors introduce multiplicative gating units
that learn when to read from and write to the memory cell.

### LSTM architecture (original 1997 version)

The original LSTM unit consists of:

**Memory cell c(t)** — the Constant Error Carousel
Self-recurrent linear unit storing the cell state across
time steps with constant weight = 1.0.

**Input gate i(t)** — multiplicative unit
Controls whether new information is allowed to enter the
memory cell. Activation = sigmoid(W_i · [x(t), h(t-1)] + b_i).
When close to 0, the cell content is protected from
irrelevant inputs.

**Output gate o(t)** — multiplicative unit
Controls whether the memory cell content is exposed to
the rest of the network. Activation = sigmoid(W_o · [x(t),
h(t-1)] + b_o). When close to 0, other units are protected
from currently irrelevant memory contents.

The forget gate was NOT in the original 1997 paper — it
was added by Gers, Schmidhuber & Cummins (2000) and is
now considered standard. Modern LSTM implementations
(including Keras) use the three-gate version.

### Cell state update — the key equation

c(t) = c(t-1) + g(net_c(t)) · y_in(t)

where:
- c(t-1) is the previous cell state (CEC self-recurrence)
- g(net_c(t)) is the squashed input to the cell
- y_in(t) is the input gate activation
- The addition + ensures constant error flow

Output of the memory cell:
y_c(t) = h(c(t)) · y_out(t)

where h() is typically tanh and y_out(t) is the output
gate activation.

### Why this matters for IDS

The vanishing gradient problem is precisely the limitation
that prevents standard RNNs from detecting attacks with
long temporal signatures in network traffic. Examples:

- **Slow DDoS attacks**: Connection patterns evolve over
  hundreds of flows. Standard RNN cannot link initial
  reconnaissance to later flood phase.
- **Multi-stage infiltration**: An infiltration attack
  (such as in CICIDS2017) involves multiple stages
  separated by long time gaps — initial dropbox download,
  later port scan, then exploitation.
- **Brute force patterns**: SSH-Patator generates thousands
  of failed authentication attempts before a successful one.
  LSTM can maintain memory of the failure pattern across
  the entire sequence.

### Experimental validation in the paper

Hochreiter & Schmidhuber test LSTM on six experimental
problems specifically designed to require learning long
temporal dependencies — problems where standard RNNs
fail completely:

- **Experiment 1**: Embedded Reber Grammar (long time lags)
- **Experiment 2**: Noise-free sequences with long time lags
- **Experiment 3**: Noisy sequences with long time lags
- **Experiment 4**: Adding problem (memorize and add two
  numbers separated by many time steps)
- **Experiment 5**: Multiplication problem
- **Experiment 6**: Temporal order of widely separated inputs

In all experiments, LSTM successfully learns dependencies
spanning 1000+ time steps where standard RNNs, BPTT, and
RTRL fail to converge.

### Properties of LSTM as documented in the paper

| Property | Value |
|---|---|
| Learns long-range dependencies | ✅ Up to 1000+ time steps |
| Vanishing gradient problem | ✅ Solved by CEC |
| Exploding gradient problem | ✅ Mitigated by gating |
| Computational complexity per step | O(1) — same as RNN |
| Memory footprint | Larger than RNN (multiple gates) |
| Training time | Slower per epoch than RNN |
| Convergence | Faster on long-dependency tasks |

### How modern LSTM differs from the 1997 paper

Modern implementations (Keras, PyTorch) include:
- **Forget gate** (Gers et al. 2000) — explicit mechanism
  to clear cell state
- **Peephole connections** (Gers & Schmidhuber 2000) —
  gates can see the cell state directly
- **Coupled input-forget gates** (some variants)
- **Layer normalization** (modern addition)

For notebook 05, the Keras `LSTM` layer uses the modern
three-gate version with forget gate by default.

### Recommended LSTM hyperparameters for notebook 05

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(128, return_sequences=True,
         input_shape=(timesteps, n_features)),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # binary classification
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)
```

Hyperparameter rationale based on Hochreiter & Schmidhuber:
- **128 units in first layer**: large enough to capture
  diverse temporal patterns in network flows
- **64 units in second layer**: forces hierarchical
  abstraction (low-level → high-level features)
- **Dropout 0.2**: prevents overfitting on the
  imbalanced CICIDS2017 dataset
- **tanh activation (default)**: keeps cell state bounded,
  prevents value explosion across long sequences

### How this paper will be cited in the report
- Introduction of LSTM as third model (notebook 05)
- Theoretical justification for handling long temporal
  dependencies in network traffic
- Explanation of why standard RNNs fail on IDS sequences
- Description of the Constant Error Carousel and gating
  mechanism in the Methodology section
- Foundational reference for the deep learning component
  of the hybrid IDS

### Key quote for report
Hochreiter & Schmidhuber (1997) introduce Long Short-Term
Memory networks specifically to address the vanishing
gradient problem that prevents standard recurrent neural
networks from learning temporal dependencies spanning
more than approximately ten time steps. The Constant
Error Carousel mechanism enables LSTM to maintain
memory across arbitrary time lags — a property essential
for detecting multi-stage network attacks where attack
signatures span hundreds of consecutive network flows.

### Limitations noted by the authors
- LSTM has more parameters than standard RNN (gates require
  additional weight matrices) — increases training time
- Sequential nature limits parallelization across time
  steps (unlike CNN or Transformer)
- The original 1997 paper does not include forget gate —
  added later by Gers et al. (2000)
- Difficult to interpret what each memory cell encodes
  (similar to other deep learning black-box concerns)


---

## Summary of literature foundation for the IDS project

This literature review establishes the theoretical and
empirical foundation for the proposed hybrid intrusion
detection system:

- **Dataset choice** justified by Sharafaldin et al. (2018)
  and Ring et al. (2019) — CICIDS2017 is the most complete
  publicly available IDS benchmark dataset.
- **Random Forest** as baseline ML model — justified by
  Breiman (2001) for its convergence guarantees and by
  Sharafaldin et al. (2018) for its F1=0.97 performance
  on CICIDS2017.
- **XGBoost** as second model — justified by Chen & Guestrin
  (2016) for its regularized objective, sparsity-aware
  algorithm, and 42x speed advantage over standard GBM.
- **LSTM** as deep learning model — justified by Hochreiter
  & Schmidhuber (1997) for its ability to learn long-range
  temporal dependencies, and by Ferrag et al. (2020) for
  its 97%+ accuracy on modern IDS benchmarks.
- **Hybrid architecture** (ML + Snort signatures) — addresses
  a research gap identified across all surveyed papers,
  none of which propose a unified hybrid system with a
  Fusion Engine.