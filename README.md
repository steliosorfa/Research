# Drug Response Prediction (DRP) — Πτυχιακή Εργασία

**Φοιτητής:** Στυλιανός Ορφανίδης  
**Επιβλέπων:** Χρίστος Δίου  
**Ίδρυμα:** Χαροκόπειο Πανεπιστήμιο, 2026  

---

## Περιγραφή

Η εργασία μελετά συστηματικά το πρόβλημα **Πρόβλεψης Απόκρισης Φαρμάκων** (Drug Response Prediction, DRP) — την πρόβλεψη της ευαισθησίας καρκινικών κυτταρικών σειρών σε αντικαρκινικά φάρμακα μέσω νευρωνικών δικτύων.

Οι κύριοι ερευνητικοί άξονες είναι:

- **Αναπαράσταση φαρμάκων:** Συστηματική σύγκριση ECFP fingerprints, 2D μοριακών γράφων και 3D γεωμετρικών γράφων (A2A + B2B με γωνίες δεσμών από 3D conformers)
- **Αναπαράσταση κυτταρικών σειρών:** Χρήση αποκλειστικά gene expression δεδομένων (top-1000 γονίδια βάσει διασποράς) — αντιμετώπιση του προβλήματος διαστατικότητας (curse of dimensionality) χωρίς την ανάγκη multi-omics δεδομένων
- **Σενάρια γενίκευσης:** Αξιολόγηση υπό τρία σενάρια (random, blind drug, blind cell) για αξιόπιστη εκτίμηση της ικανότητας γενίκευσης σε νέα φάρμακα και κυτταρικές σειρές

---

## Dataset

**GDSC1** (Genomics of Drug Sensitivity in Cancer) μέσω TDC  
- 177.310 ζεύγη φαρμάκου-κυτταρικής σειράς  
- 223 φάρμακα, 948 κυτταρικές σειρές  
- Target: ln(IC50) — raw, χωρίς κανονικοποίηση

---

## Δομή Repository

```
tdc_drugres_baseline/
│
├── src/
│   ├── baseline_1/
│   │   └── baseline1.py              # MLP + ECFP fingerprints
│   │
│   ├── baseline_2/
│   │   ├── baseline2_gat.py          # Graph Attention Network (GAT)
│   │   ├── baseline2_gcn.py          # Graph Convolutional Network (GCN)
│   │   └── baseline2_gine.py         # Graph Isomorphism Network with Edges (GINE)
│   │
│   └── 3d_baseline/
│       └── 3d_baseline_tuned.py      # Τελικό μοντέλο: 3D GeoGNN (A2A + B2B)
│
└── results/                          # Αποτελέσματα εκπαίδευσης (τοπικά, εκτός repo)
```

---

## Μοντέλα

### Baseline 1 — MLP + ECFP (`baseline1.py`)
Κωδικοποίηση φαρμάκων με **Extended Connectivity Fingerprints** (ECFP4, radius=2, 2048 bits) και επεξεργασία μέσω MLP. Χρησιμοποιείται ως ισχυρό μοντέλο αναφοράς.

### Baseline 2 — 2D Graph Neural Networks (`baseline2_*.py`)
Κωδικοποίηση φαρμάκων ως **2D μοριακοί γράφοι** (άτομα=κόμβοι, δεσμοί=ακμές). Τρεις παραλλαγές:
- **GAT:** Μηχανισμός attention για διαφορετική στάθμιση γειτόνων
- **GCN:** Κανονικοποιημένη συνέλιξη γράφου
- **GINE:** Άθροισμα χωρίς κανονικοποίηση + edge features, μέγιστη εκφραστική ισχύς

### Τελικό Μοντέλο — 3D GeoGNN (`3d_baseline_tuned.py`)
Κωδικοποίηση φαρμάκων με **τρισδιάστατους γεωμετρικούς γράφους** μέσω δύο παράλληλων streams:
- **A2A (Atom-to-Atom):** Κλασικός μοριακός γράφος με μήκη δεσμών από 3D conformers
- **B2B (Bond-to-Bond):** Γράφος όπου οι δεσμοί γίνονται κόμβοι και οι γωνίες δεσμών αποτελούν χαρακτηριστικά ακμών

---

## Ενιαίο Πειραματικό Πλαίσιο

Όλα τα μοντέλα εκπαιδεύτηκαν με κοινές ρυθμίσεις:

| Παράμετρος | Τιμή |
|---|---|
| Seed | 44 |
| Split | 80 / 10 / 10 (train/val/test) |
| Top-K γονίδια | 1.000 (βάσει διασποράς από train only) |
| Standardization | Z-score (από train only) |
| Target | ln(IC50) — raw |
| Loss | MSE |
| Optimizer | Adam |
| Early stopping patience | 20 epochs |
| Max epochs | 80 |
| Gradient clipping | max_norm=5.0 |

### Split Scenarios
- **Random:** Τυχαίος διαχωρισμός — αξιολόγηση παρεμβολής
- **Blind Drug:** GroupShuffleSplit ανά φάρμακο — γενίκευση σε νέα φάρμακα
- **Blind Cell:** GroupShuffleSplit ανά κυτταρική σειρά — γενίκευση σε νέες κυτταρικές σειρές

---

## Αποτελέσματα

| Μοντέλο | Random RMSE/PCC | Blind Drug RMSE/PCC | Blind Cell RMSE/PCC |
|---------|-----------------|---------------------|---------------------|
| MLP+ECFP | **0.9724 / 0.9339** | 2.2928 / 0.5599 | 1.3214 / 0.8757 |
| GAT | 1.7642 / 0.7733 | 2.1983 / 0.5804 | 1.8533 / 0.7466 |
| GCN | 1.2545 / 0.8873 | 2.6733 / 0.3767 | 1.3970 / 0.8598 |
| GINE | 1.3205 / 0.8911 | 2.1900 / 0.5671 | 1.5065 / 0.8529 |
| **3D GeoGNN** | 1.2163 / 0.9145 | **1.8613 / 0.7020** | **1.4928 / 0.8620** |

---

## Dependencies

```bash
pip install torch torch-geometric
pip install rdkit
pip install PyTDC
pip install scikit-learn numpy matplotlib
```

---

## Εκτέλεση

```bash
# Αλλαγή split strategy μέσα στο αρχείο:
# cfg.split_type = "random" | "blind_drug" | "blind_cell"

python src/baseline_1/baseline1.py
python src/baseline_2/baseline2_gine.py
python src/3d_baseline/3d_baseline_tuned.py
```