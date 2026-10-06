"""E229 independent check (written without reading 01_tables.py or p8common.py).
Recomputes T1 from raw forms.csv, and XGBoost rows of T2/T3 from the STORED
March-2026 feature matrix. Compares to E229 outputs where they exist."""
import json, re, io, os
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb

ROOT = r"D:\documents\volcarch-repo\experiments"
RES = os.path.join(ROOT, "E229_p8_revision_tables", "results")
LOG = io.open(os.path.join(RES, "independent_check_log.txt"), "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.write(s + "\n")
    LOG.flush()


LISTS = [("27", "Muna"), ("48", "Bugis"), ("166", "Makassar"), ("192", "Wolio"), ("226", "Toraja-Sadan"), ("674", "Tolaki")]

# ---------- (a) T1 from raw ----------
forms = pd.read_csv(os.path.join(ROOT, "E022_linguistic_subtraction", "data", "abvd", "cldf", "forms.csv"),
                    dtype=str, keep_default_na=False)


def clean(v):
    v = re.sub(r"\[[^\]]*\]", "", v)
    return v.strip(" \t.,;:!?-'\"")


def has_form(r):
    v = r["Value"] if r["Value"].strip() else r["Form"]
    return clean(v) != ""


sub = forms[forms["Language_ID"].isin([l for l, _ in LISTS])]
sub = sub[sub.apply(has_form, axis=1)].copy()
sub["coded"] = sub["Cognacy"].str.strip() != ""
t1 = {}
for lid, nm in LISTS:
    s = sub[sub["Language_ID"] == lid]
    t1[nm] = dict(forms=int(len(s)), coded=int(s.coded.sum()), candidate=int((~s.coded).sum()))
t1["TOTAL"] = dict(forms=int(len(sub)), coded=int(sub.coded.sum()), candidate=int((~sub.coded).sum()))
for k, v in t1.items():
    v["pct_coded"] = round(100 * v["coded"] / v["forms"], 1)
    v["pct_candidate"] = round(100 * v["candidate"] / v["forms"], 1)
    say("T1", k, v)

# ---------- (b) features ----------
fm = pd.read_csv(os.path.join(ROOT, "E027_ml_substrate_detection", "data", "features_matrix.csv"), keep_default_na=False)
say("matrix rows", len(fm), "label sum", fm.label.sum(), "languages", fm.language.value_counts().to_dict())
say("initial_char values", sorted(fm.initial_char.unique()), "semantic_domain", sorted(fm.semantic_domain.unique()))
BASE = ["form_length", "n_vowels", "vowel_ratio", "ends_in_vowel", "has_glottal", "has_nasal_cluster",
        "has_reduplication", "n_consonant_clusters", "has_prefix_like"]
INIT = ["init_a", "init_b", "init_k", "init_m", "init_other", "init_p", "init_s", "init_t"]
SEM = ["sem_ACTION", "sem_BODY", "sem_GRAMMAR", "sem_NATURE", "sem_NUMBER", "sem_OTHER", "sem_QUALITY"]


def build(df):
    X = pd.DataFrame(index=df.index)
    for c in BASE:
        X[c] = pd.to_numeric(df[c]).astype(float)
    for c in INIT:
        X[c] = (df["initial_char"] == c[5:]).astype(float)
    X["is_core_vocab"] = pd.to_numeric(df["is_core_vocab"]).astype(float)
    for c in SEM:
        X[c] = (df["semantic_domain"] == c[4:]).astype(float)
    assert list(X.columns) == BASE + INIT + ["is_core_vocab"] + SEM and X.shape[1] == 25
    return X.values, X.values[:, :17]


def clf():
    return xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, reg_lambda=1.0,
                             scale_pos_weight=1.0, eval_metric="logloss", random_state=42, verbosity=0)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(y, p), accuracy=accuracy_score(y, pred),
                f1_cand=f1_score(y, pred, pos_label=0, zero_division=0),
                prec_cand=precision_score(y, pred, pos_label=0, zero_division=0),
                rec_cand=recall_score(y, pred, pos_label=0, zero_division=0),
                f1_coded=f1_score(y, pred, pos_label=1, zero_division=0),
                always_coded=float((y == 1).mean()))


def cv(X, y):
    rows, seedmeans = [], []
    for seed in range(10):
        sk = StratifiedKFold(5, shuffle=True, random_state=seed * 7 + 13)
        sr = []
        for tr, te in sk.split(X, y):
            m = clf().fit(X[tr], y[tr])
            sr.append(metrics(y[te], m.predict_proba(X[te])[:, 1]))
        rows += sr
        seedmeans.append({k: np.mean([r[k] for r in sr]) for k in sr[0]})
    out = {}
    for k in rows[0]:
        out[k] = dict(mean=float(np.mean([r[k] for r in rows])),
                      sd_seed_means=float(np.std([s[k] for s in seedmeans], ddof=1)),
                      sd_folds=float(np.std([r[k] for r in rows], ddof=1)))
    return out


def lolo(X, y, lang):
    out = {}
    for _, nm in LISTS:
        te = (lang == nm).values
        tr = ~te
        m = clf().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        yt = y[te]
        pred = (p >= 0.5).astype(int)
        out[nm] = dict(n=int(te.sum()), n_cand=int((yt == 0).sum()), auc=float(roc_auc_score(yt, p)),
                       accuracy=float(accuracy_score(yt, pred)),
                       majority_acc=float(max((yt == 1).mean(), (yt == 0).mean())),
                       always_coded=float((yt == 1).mean()),
                       f1_cand=float(f1_score(yt, pred, pos_label=0, zero_division=0)))
    return out


def run(df, tag):
    y = df["label"].astype(int).values
    X25, X17 = build(df)
    r = {"order": tag, "cv": {"FM25": cv(X25, y), "F17": cv(X17, y)},
         "lolo": {"FM25": lolo(X25, y, df["language"]), "F17": lolo(X17, y, df["language"])}}
    for s in ("FM25", "F17"):
        say(tag, s, "CV", {k: round(v["mean"], 4) for k, v in r["cv"][s].items()},
            "SDseed", round(r["cv"][s]["auc"]["sd_seed_means"], 4), "SDfold", round(r["cv"][s]["auc"]["sd_folds"], 4))
        say(tag, s, "LOLO", {k: round(v["auc"], 4) for k, v in r["lolo"][s].items()},
            "mean", round(np.mean([v["auc"] for v in r["lolo"][s].values()]), 4))
    return r


fm_stored = fm.copy()
order = {i: k for k, i in enumerate(forms["ID"])}
missing = [i for i in fm.form_id if i not in order]
say("matrix form_ids not found in forms.csv:", len(missing))
fm_forms = fm.assign(_o=fm.form_id.map(order)).sort_values("_o").reset_index(drop=True)
same_order = bool((fm.form_id.values == fm_forms.form_id.values).all())
say("stored order identical to forms.csv order:", same_order)
raw_lab = sub.set_index("ID")["coded"].astype(int)
mm = fm.set_index("form_id")["label"].astype(int)
common = mm.index.intersection(raw_lab.index)
say("matrix ids", len(mm), "in raw-kept ids", len(common), "label mismatches", int((mm[common] != raw_lab[common]).sum()),
    "raw ids not in matrix", len(set(raw_lab.index) - set(mm.index)))

R_stored = run(fm_stored, "stored_order")
R_forms = run(fm_forms, "forms_csv_order") if not same_order else None

# ---------- (c) compare ----------
csvs = {n: os.path.exists(os.path.join(RES, n)) for n in ("T1_label_by_list.csv", "T2_cv_performance.csv", "T3_lolo.csv")}
say("E229 CSVs present:", csvs)
log = open(os.path.join(RES, "run_log.txt"), encoding="utf-8").read()


def grab(block_label):
    m = re.search(re.escape(block_label) + r" XGBoost \{(.*?)\}", log)
    return {k: float(v) for k, v in re.findall(r"'(\w+)': np\.float64\(([\d.]+)\)", m.group(1))} if m else {}


anc = {c["name"]: c for c in json.load(open(os.path.join(RES, "anchor_checks.json")))["checks"]}
cmp = []


def add(q, e, mine, tol, src):
    if e is None:
        v, d = "NOT COMPARABLE (E229 value absent)", None
    else:
        d = abs(e - mine)
        v = "AGREE" if d <= tol else "DISAGREE"
    cmp.append(dict(quantity=q, e229=e, independent=mine, abs_diff=d, verdict=v, e229_source=src))


for nm in [n for _, n in LISTS] + ["TOTAL"]:
    for k in ("forms", "coded", "candidate"):
        add(f"T1 {nm} {k}", None, t1[nm][k], 0, "T1 csv missing")
# orchestrator's edit, 2026-10-06: after DESIGN amendment A1 the anchor is a dict keyed by list name
anc_c = anc["candidates_per_list (keyed by list; amendment A1)"]["got"]
for (_, nm) in LISTS:
    g = anc_c[nm]
    add(f"T1 {nm} candidates (vs anchor_checks.json 'got', order Muna,Bugis,Makassar,Wolio,Tae',Tolaki)", g,
        t1[nm]["candidate"], 0, "anchor_checks.json")
mp = {"auc_mean": "auc", "accuracy_mean": "accuracy", "f1_candidate_mean": "f1_cand",
      "precision_candidate_mean": "prec_cand", "recall_candidate_mean": "rec_cand", "f1_coded_mean": "f1_coded"}
for s in ("FM25", "F17"):
    g = grab(s)
    for k, mk in mp.items():
        add(f"T2 {s} XGB {k}", g.get(k), R_stored["cv"][s][mk]["mean"], 0.0005, "run_log.txt (4 dp)")
    a = anc[f"cv_auc_{s}_xgb"]
    add(f"T2 {s} XGB auc (anchor json, unrounded)", a["got"], R_stored["cv"][s]["auc"]["mean"], 0.0005, "anchor_checks.json")
    a = anc[f"cv_auc_sd_seed_means_{s}_xgb"]
    add(f"T2 {s} XGB auc SD of seed means", a["got"], R_stored["cv"][s]["auc"]["sd_seed_means"], 0.0005, "anchor_checks.json")
    for k in ("accuracy", "f1_cand", "prec_cand", "rec_cand", "f1_coded", "auc"):
        add(f"T2 {s} XGB {k} SD of 50 folds", None, R_stored["cv"][s][k]["sd_folds"], 0, "csv missing")
    add(f"T2 {s} always-coded accuracy", None, R_stored["cv"][s]["always_coded"]["mean"], 0, "csv missing")
    for _, nm in LISTS:
        a = anc[f"lolo_auc_{s}_{nm}_vs_S7 (3 decimals)"]
        add(f"T3 {s} {nm} AUC (3 dp)", a["got"], round(R_stored["lolo"][s][nm]["auc"], 3), 0.0005, "anchor_checks.json")
        for k in ("n", "n_cand", "accuracy", "majority_acc", "always_coded", "f1_cand"):
            add(f"T3 {s} {nm} {k}", None, R_stored["lolo"][s][nm][k], 0, "T3 csv missing")
    a = anc[f"lolo_mean_auc_{s}"]
    add(f"T3 {s} mean AUC", a["got"], float(np.mean([v["auc"] for v in R_stored["lolo"][s].values()])), 0.0005, "anchor_checks.json")

comp = [c for c in cmp if c["abs_diff"] is not None and "SD" not in c["quantity"] and "candidates (vs" not in c["quantity"]]
maxd = max(c["abs_diff"] for c in comp)
say("max abs diff among comparable means:", maxd)
for c in cmp:
    if c["verdict"] == "DISAGREE":
        say("DISAGREE:", c)

out = dict(csv_presence=csvs, t1_independent=t1, comparisons=cmp, max_abs_diff_in_means=maxd,
           row_order=dict(stored_order_identical_to_forms_csv_order=same_order, matrix_ids_missing_from_forms=len(missing)),
           independent_stored_order=R_stored, independent_forms_csv_order=R_forms)
json.dump(out, open(os.path.join(RES, "independent_check.json"), "w", encoding="utf-8"), indent=1, default=float)
say("written independent_check.json")
