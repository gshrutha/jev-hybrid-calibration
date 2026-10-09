"""
Calibration baselines and a hybrid (Jev + structured fields) model for the
jev-ood-calibration synthetic tickets. All results are out-of-fold
(k-fold cross-validation over tickets).

  raw             Jev's probabilities as returned
  global T        one temperature for all question types
  per-type T      one temperature per question type
  hybrid (Jev)    logistic regression on Jev's three answers for the ticket
  hybrid (+meta)  the same plus customer_tier and channel

Usage:
  python scripts/hybrid_calib.py --data data/val.jsonl --results results/my_run.jsonl
"""
import argparse
import json

import numpy as np
import torch

EPS = 1e-4    # clip probabilities before taking logs (Jev returns exact zeros)
N_BINS = 15   # same ECE binning as eval_dump


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_dist(vec):
    v = np.atleast_1d(np.asarray(vec, dtype=float))
    if v.size == 1:  # a single probability -> [no, yes]
        v = np.array([1.0 - v[0], v[0]])
    return v


def clip_norm(P):
    P = np.clip(P, EPS, 1.0)
    return P / P.sum(axis=-1, keepdims=True)


def temp_scale(P, T):
    L = np.log(clip_norm(P)) / T
    L -= L.max(axis=-1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=-1, keepdims=True)


def total_nll(P, y):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0)).sum()


def fit_T(groups):
    """Temperature minimising total NLL over a list of (P, y) groups."""
    grid = np.exp(np.linspace(np.log(0.1), np.log(20.0), 300))
    losses = [sum(total_nll(temp_scale(P, T), y) for P, y in groups) for T in grid]
    return float(grid[int(np.argmin(losses))])


def to_cols(P, y):
    return np.column_stack([P.max(axis=1),
                            (P.argmax(axis=1) == y).astype(float),
                            P[np.arange(len(y)), y]])


def summarize(cols):
    conf, correct, ptrue = cols[:, 0], cols[:, 1], cols[:, 2]
    bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    ece = sum(np.mean(bins == b) * abs(conf[bins == b].mean() - correct[bins == b].mean())
              for b in np.unique(bins))
    nll = -np.log(np.clip(ptrue, 1e-12, 1.0)).mean()
    return len(conf), correct.mean(), nll, ece


def fit_logreg(X, y, n_classes, l2):
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.long)
    W = torch.zeros(X.shape[1], n_classes, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(n_classes, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach(), b.detach()


def predict_logreg(W, b, X):
    X = torch.tensor(X, dtype=torch.float64)
    return torch.softmax(X @ W + b, dim=-1).numpy()


def norm(v):
    s = str(v).strip().lower()
    return {"true": "yes", "false": "no"}.get(s, s)


def align(data, results):
    if len(data) != len(results):
        raise SystemExit(f"row count mismatch: {len(data)} inputs vs {len(results)} results")
    rows, bad = [], []
    for i, (d, r) in enumerate(zip(data, results)):
        if r.get("probs") is None or r.get("target") is None:
            raise SystemExit(f"row {i} has no probabilities (failed call?): {r}")
        if d["type"] != r["type"] or norm(d["label"]) != norm(r.get("gold")):
            bad.append(i)
        rows.append({"type": r["type"],
                     "P": as_dist(r["probs"]),
                     "y": int(np.argmax(as_dist(r["target"]))),
                     "state": d["state"],
                     "key": json.dumps(d["state"], sort_keys=True)})
    if bad:
        i = bad[0]
        raise SystemExit(
            f"{len(bad)} rows don't line up with the input file. First one, row {i}:\n"
            f"  input:  type={data[i]['type']} label={data[i]['label']}\n"
            f"  result: type={results[i]['type']} gold={results[i].get('gold')}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/val.jsonl")
    ap.add_argument("--results", default="results/my_run.jsonl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--l2", type=float, default=1e-2)
    args = ap.parse_args()

    rows = align(load_jsonl(args.data), load_jsonl(args.results))
    types = sorted({r["type"] for r in rows})
    if "score" not in types:
        raise SystemExit("no 'score' questions found")

    # group the questions that belong to the same ticket: consecutive rows
    # with the same state (two tickets can have identical text, so the
    # state alone is not a unique id)
    groups = []
    for i, r in enumerate(rows):
        if (groups and rows[groups[-1][0]]["key"] == r["key"]
                and r["type"] not in {rows[j]["type"] for j in groups[-1]}):
            groups[-1].append(i)
        else:
            groups.append([i])
    tickets = {}
    for n, g in enumerate(groups):
        t = {rows[j]["type"]: j for j in g}
        if set(t) != set(types) or len(g) != len(types):
            raise SystemExit(f"ticket {n} (rows {g}) does not have exactly one question of each type")
        tickets[f"t{n:04d}"] = t
    keys = sorted(tickets)

    rng = np.random.default_rng(args.seed)
    fold_of = {str(k): f for k, f in zip(rng.permutation(keys), np.arange(len(keys)) % args.folds)}

    first = [rows[t["score"]]["state"] for t in tickets.values()]
    tiers = sorted({str(s.get("customer_tier")) for s in first})
    channels = sorted({str(s.get("channel")) for s in first})

    def onehot(value, vocab):
        v = np.zeros(len(vocab))
        v[vocab.index(str(value))] = 1.0
        return v

    def features(k, with_meta):
        t = tickets[k]
        parts = [np.log(clip_norm(rows[t[ty]]["P"])) for ty in types]
        if with_meta:
            s = rows[t["score"]]["state"]
            parts += [onehot(s.get("customer_tier"), tiers), onehot(s.get("channel"), channels)]
        return np.concatenate(parts)

    def stack(ks, ty):
        P = np.stack([rows[tickets[k][ty]]["P"] for k in ks])
        y = np.array([rows[tickets[k][ty]]["y"] for k in ks])
        return P, y

    n_levels = rows[tickets[keys[0]]["score"]]["P"].size
    store, temps = {}, {ty: [] for ty in types}

    def add(method, ty, P, y):
        store.setdefault(method, {}).setdefault(ty, []).append(to_cols(P, y))

    for f in range(args.folds):
        tr = [k for k in keys if fold_of[k] != f]
        te = [k for k in keys if fold_of[k] == f]

        train = {ty: stack(tr, ty) for ty in types}
        T_global = fit_T(list(train.values()))
        for ty in types:
            P, y = stack(te, ty)
            T_ty = fit_T([train[ty]])
            temps[ty].append(T_ty)
            add("raw", ty, clip_norm(P), y)
            add("global T", ty, temp_scale(P, T_global), y)
            add("per-type T", ty, temp_scale(P, T_ty), y)

        y_tr, y_te = train["score"][1], stack(te, "score")[1]
        for name, meta in [("hybrid (Jev)", False), ("hybrid (+meta)", True)]:
            X_tr = np.stack([features(k, meta) for k in tr])
            X_te = np.stack([features(k, meta) for k in te])
            mu, sd = X_tr.mean(0), X_tr.std(0)
            sd[sd == 0] = 1.0
            W, b = fit_logreg((X_tr - mu) / sd, y_tr, n_levels, args.l2)
            add(name, "score", predict_logreg(W, b, (X_te - mu) / sd), y_te)

    combo = {ty: store["per-type T"][ty] for ty in types if ty != "score"}
    combo["score"] = store["hybrid (+meta)"]["score"]
    store["per-type T + hybrid (+meta)"] = combo

    print(f"{args.folds}-fold cross-validation over {len(keys)} tickets (seed {args.seed})")
    print("per-type temperature, mean over folds: "
          + ", ".join(f"{ty} {np.mean(v):.2f}" for ty, v in temps.items()))
    print(f"\n{'method':30s} {'type':8s} {'n':>5s} {'acc':>7s} {'NLL':>7s} {'ECE':>7s}")
    for method, by_type in store.items():
        groups = {ty: np.concatenate(v) for ty, v in by_type.items()}
        if len(groups) > 1:
            groups["all"] = np.concatenate(list(groups.values()))
        for ty, cols in groups.items():
            n, acc, nll, ece = summarize(cols)
            print(f"{method:30s} {ty:8s} {n:5d} {acc:7.3f} {nll:7.3f} {ece:7.3f}")
        print()


if __name__ == "__main__":
    main()
