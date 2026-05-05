# Literature Review Notes — IDS-KMUTT Hybrid IDS Project

**Author:** Darren Touopi  
**Programme:** BAC+4 — Sécurité et Qualité des Réseaux (SQR), Polytech Dijon  
**Last updated:** 30/04/2026  

---

## Reading List

| # | Paper | Authors | Year | Status |
|---|---|---|---|---|
| 1 | Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization | Sharafaldin, Lashkari, Ghorbani | 2018 | ✅ Read |
| 2 | A Survey of Network-based Intrusion Detection Data Sets | Ring, Wunderlich, Scheuring, Landes, Hotho | 2019 | ✅ Read |
| 3 | Deep Learning for Cyber Security Intrusion Detection | Ferrag et al. | 2020 | ⏳ To read |
| 4 | Random Forests | Breiman | 2001 | ⏳ To read |
| 5 | XGBoost: A Scalable Tree Boosting System | Chen & Guestrin | 2016 | ⏳ To read |
| 6 | Long Short-Term Memory | Hochreiter & Schmidhuber | 1997 | ⏳ To read |

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