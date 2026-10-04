#!/usr/bin/env python3
"""
Science Olympiad event-group builder
====================================

Splits the club's events into study groups (default: five groups of 3 events
and two groups of 4) so that events with correlated performance land together,
build events lean on preference instead of performance, and every group can be
anchored by about 2 officers ("GREAT" members).

Model
-----
1. Event-to-event similarity matrix S, blending three signals:
   - Performance correlation. Test difficulty and student ability are fitted
     together (score = overall + ability - difficulty), so harder tests are
     corrected for without being fooled by which students chose them. Scores
     are then standardized per event, and each student's general ability (estimated from their OTHER events, so it does
     not leak into the pair being measured) is removed. The residuals are
     correlated over students who took both events.
   - Preference affinity: cosine similarity of the ranked preferences
     (each student lists 0-4 events; 1st choice = 4 pts, 2nd = 3, 3rd = 2,
     4th = 1).
   - Co-selection affinity: cosine similarity of which tests students chose
     to take.
   Each signal is standardized, then blended. The performance weight depends
   on event type (test, hybrid or build) and is scaled down when
   few students took both tests.

2. Constrained partition. The objective is total within-group similarity
   plus officer-to-group fit, minus a penalty when a group's officer count is
   off target. Group sizes are exact. Search is an iterated local search over
   event swaps with many random restarts; for each candidate partition the
   officer placement is solved exactly as an assignment problem. With 23
   events this reliably finds the optimum, and the output reports how many
   restarts reached the same best answer.

3. Bootstrap stability: students are resampled with replacement, S is rebuilt
   and the partition re-solved many times. The share of runs in which two
   events end up together tells you how much to trust each pairing.

4. Student placement (exact MILP): every student goes to one group, within
   size limits, with about 2 officers per group and a balanced, slightly
   top-leaning mix of provisional A/B/C teams. Officers rank ahead of
   non-officers when teams are drawn. Students listed in LOCKED_STUDENTS
   are only placed in a group containing one of their locked events.

5. Placement test: preference and performance alignment of the final
   placement, compared against thousands of random placements.

All tunable rules and weights are in the CONFIGURATION section below.

Usage
-----
    pip install numpy scipy
    python scioly_groups.py                  # reads scioly_data.json next to this script
    python scioly_groups.py path/to/data.json

Results print to the terminal and are saved to a "results" folder next to the
data file.
"""

import csv
import difflib
import unicodedata
import json
import os
import sys
import time
from itertools import combinations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix

# ==========================================================================
#                      CONFIGURATION: EDIT ANYTHING HERE
# Every rule and weight the model uses lives in this section. The data file
# only supplies events and students; a "settings" block in the JSON is
# ignored unless USE_JSON_SETTINGS is True.
# ==========================================================================
USE_JSON_SETTINGS = False

# ---- Event groups --------------------------------------------------------
GROUP_SIZES = [3, 3, 3, 3, 3, 4, 4]      # events per group; must sum to # events

# Cannot-link time blocks: tests in the same block run at the same time, so no
# two events from one block may share a group. Unlisted events are free.
# Names are matched ignoring case, spaces and punctuation.
CANNOT_LINK_BLOCKS = {
    "A": ["Anatomy and Physiology", "Astronomy", "Chemistry Lab", "Codebusters"],
    "B": ["Botany", "Dynamic Planet", "Forensics"],
    "C": ["Designer Genes", "Remote Sensing", "Write-it/Do-It"],
    "D": ["Protein Modeling", "Rocks and Minerals", "Water Quality"],
}
NEVER_TOGETHER = []          # extra pairs, e.g. [["Boomilever", "Wright Stuff"]]
MUST_TOGETHER = []           # e.g. [["Anatomy and Physiology", "Disease Detectives"]]
MAX_BUILDS_PER_GROUP = None  # e.g. 2 (hybrids count as builds); None = no limit
AVOID_SAME_TIME_BLOCK = False  # also use each event's optional "time_block" field

# ---- Preferences ---------------------------------------------------------
# Each student lists 0-4 events, best first. Earlier = higher priority.
PREFERENCE_RANK_POINTS = [4, 3, 2, 1]        # 1st..4th choice, for event similarity
PREFERENCE_FIT_BY_RANK = [1.0, 0.8, 0.6, 0.4]  # 1st..4th choice, for student fit
PREFERENCE_INDEX_BASE = 1    # if preferences are numbers: 1 = first event listed

# ---- Performance vs. preference -------------------------------------------
# How much each event type trusts test PERFORMANCE (the rest is preference).
# A pair of events uses the geometric mean of the two.
PERF_WEIGHT_TEST = 0.65
PERF_WEIGHT_HYBRID = 0.40
PERF_WEIGHT_BUILD = 0.15
PREFERENCE_SHARE_OF_REST = 0.75  # rest of non-performance weight = co-selection
DIFFICULTY_SHRINKAGE = 2.0       # higher = more cautious calling a test hard/easy
ABILITY_SHRINKAGE = 1.0          # shrinkage on each student's general ability
MIN_OVERLAP = 4                  # fewer shared test-takers = no performance weight
FULL_CONFIDENCE_OVERLAP = 12     # shared test-takers for full performance weight
UNTAKEN_EVENT_FIT = -0.25        # fit credit for a test the student skipped
PERFORMANCE_Z_CLIP = 2.0         # cap on how far one score counts above/below average

# ---- Officers (GREAT members) ---------------------------------------------
TARGET_OFFICERS_PER_GROUP = 2
MIN_OFFICERS_PER_GROUP = 1       # hard minimum (skipped if not enough officers)
OFFICER_WEIGHT = 1.0             # how much officer fit shapes the event groups
OFFICER_DEVIATION_PENALTY = 2.0  # event-group stage: cost per officer off target
OFFICER_OVERFLOW_SLOTS = 2       # extra officers a group may take beyond target

# ---- Student placement ------------------------------------------------------
MAX_PEOPLE_PER_GROUP = 7
IDEAL_PEOPLE_BY_EVENT_COUNT = {3: 6, 4: 7}
SPECIAL_GROUP_MAX = {"Codebusters": 9}   # groups containing these get this max
# Provisional teams used to spread talent: top 15 = A, next 15 = B, rest = C
TEAM_SIZES = [15, 15]
OFFICERS_OUTRANK_NON_OFFICERS = True   # officers fill the highest teams first, so
                                       # no non-officer is ever on a higher team
                                       # than an officer in the same group
REQUIRE_EVERY_TEAM_IN_EVERY_GROUP = True
GROUP_FIT_AVERAGE_SHARE = 0.6    # student-group fit = this x average event fit
                                 # + the rest x best single event fit
STUDENT_FIT_WEIGHT = 1.0
OVERSIZE_PENALTY = 1.5           # per person above ideal
OVERSIZE_PENALTY_SPECIAL = 0.25  # per person above ideal, Codebusters group
UNDERSIZE_PENALTY = 1.0          # per person below ideal
OFFICER_BALANCE_PENALTY = 0.75   # per officer off TARGET_OFFICERS_PER_GROUP
TEAM_BALANCE_PENALTY = 0.5       # per person off the club-wide team mix
LEAN_BETTER_MULTIPLIER = 1.5     # extra cost for missing top-team members or
                                 # having extra bottom-team members
ASSIGNMENT_TIME_LIMIT_SEC = 60

# Locks: students who MUST be placed in a group containing at least one of the
# listed events. Names must match the data file (case, spaces and punctuation
# are ignored). Example:
#   LOCKED_STUDENTS = {
#       "Jane Doe":   ["Codebusters"],
#       "John Smith": ["Circuit Lab", "Thermodynamics"],  # either one is fine
#   }
LOCKED_STUDENTS = {
}

# ---- Search, stability and evaluation ---------------------------------------
SEARCH_RESTARTS = 100
SEARCH_KICKS = 10
BOOTSTRAP_RUNS = 100             # 0 to skip the stability check
BOOTSTRAP_RESTARTS = 10
BOOTSTRAP_KICKS = 5
UNCERTAIN_PLACEMENT_BELOW = 0.4  # flag events grouped together less than this
EVALUATION_RANDOM_BASELINES = 2000  # random placements to compare against
RANDOM_SEED = 42
# ==========================================================================


DEFAULT_SETTINGS = {
    "group_sizes": GROUP_SIZES,
    "preference_index_base": PREFERENCE_INDEX_BASE,
    "preference_rank_points": PREFERENCE_RANK_POINTS,
    "perf_weight_test": PERF_WEIGHT_TEST,
    "perf_weight_hybrid": PERF_WEIGHT_HYBRID,
    "perf_weight_build": PERF_WEIGHT_BUILD,
    "difficulty_shrinkage": DIFFICULTY_SHRINKAGE,
    "preference_share_of_rest": PREFERENCE_SHARE_OF_REST,
    "min_overlap": MIN_OVERLAP,
    "full_confidence_overlap": FULL_CONFIDENCE_OVERLAP,
    "ability_shrinkage": ABILITY_SHRINKAGE,
    "target_officers_per_group": TARGET_OFFICERS_PER_GROUP,
    "officer_weight": OFFICER_WEIGHT,
    "officer_deviation_penalty": OFFICER_DEVIATION_PENALTY,
    "min_officers_per_group": MIN_OFFICERS_PER_GROUP,
    "untaken_event_fit": UNTAKEN_EVENT_FIT,
    "never_together": NEVER_TOGETHER,
    "must_together": MUST_TOGETHER,
    "avoid_same_time_block": AVOID_SAME_TIME_BLOCK,
    "max_builds_per_group": MAX_BUILDS_PER_GROUP,
    "student_fit_weight": STUDENT_FIT_WEIGHT,
    "oversize_penalty": OVERSIZE_PENALTY,
    "oversize_penalty_special": OVERSIZE_PENALTY_SPECIAL,
    "undersize_penalty": UNDERSIZE_PENALTY,
    "officer_balance_penalty": OFFICER_BALANCE_PENALTY,
    "tier_balance_penalty": TEAM_BALANCE_PENALTY,
    "assignment_time_limit_sec": ASSIGNMENT_TIME_LIMIT_SEC,
    "search_restarts": SEARCH_RESTARTS,
    "search_kicks": SEARCH_KICKS,
    "bootstrap_runs": BOOTSTRAP_RUNS,
    "bootstrap_restarts": BOOTSTRAP_RESTARTS,
    "bootstrap_kicks": BOOTSTRAP_KICKS,
    "random_seed": RANDOM_SEED,
}


def _norm(name):
    """Lowercase letters and digits only, after folding look-alike characters
    (accents, full-width letters, ligatures) and dropping invisible ones."""
    text = unicodedata.normalize("NFKD", str(name)).casefold()
    return "".join(ch for ch in text if ch.isascii() and ch.isalnum())


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    settings = dict(DEFAULT_SETTINGS)
    json_settings = {k: v for k, v in data.get("settings", {}).items()
                     if not k.startswith("_")}
    ignored_settings = []
    if USE_JSON_SETTINGS:
        settings.update(json_settings)
    else:
        ignored_settings = sorted(k for k, v in json_settings.items()
                                  if k in settings and settings[k] != v)

    events = data["events"]
    names = [ev["name"] for ev in events]
    if len(set(names)) != len(names):
        raise ValueError("Event names must be unique.")
    name_to_idx = {n: i for i, n in enumerate(names)}
    lower_to_idx = {n.lower(): i for i, n in enumerate(names)}
    E = len(events)

    types = [str(ev.get("type", "test")).lower() for ev in events]
    for n, t in zip(names, types):
        if t not in ("test", "build", "hybrid"):
            raise ValueError(f"{n}: type must be 'test', 'build' or 'hybrid', not {t!r}.")
    is_build = np.array([t in ("build", "hybrid") for t in types])
    affinity = np.array([settings[f"perf_weight_{t}"] for t in types])
    time_blocks = [ev.get("time_block") for ev in events]

    if sum(settings["group_sizes"]) != E:
        raise ValueError(
            f"group_sizes {settings['group_sizes']} add up to "
            f"{sum(settings['group_sizes'])}, but there are {E} events.")

    students = data["students"]
    N = len(students)
    raw = np.full((N, E), np.nan)
    pref = np.zeros((N, E))
    pref_rank = np.full((N, E), 0)  # 1 = first choice, ..., 0 = not ranked
    officer = np.zeros(N, dtype=bool)
    student_names = []
    warnings = []
    base = settings["preference_index_base"]
    points = settings["preference_rank_points"]

    norm_idx = {_norm(n): i for i, n in enumerate(names)}

    def resolve_event(ref, who):
        if isinstance(ref, bool):
            raise ValueError(f"{who}: preference {ref!r} is not an event.")
        if isinstance(ref, (int, float)) and float(ref).is_integer():
            idx = int(ref) - base
            if 0 <= idx < E:
                return idx
            raise ValueError(
                f"{who}: preference {ref} is out of range (events are "
                f"{base}..{E - 1 + base} with preference_index_base={base}).")
        if isinstance(ref, str):
            if ref in name_to_idx:
                return name_to_idx[ref]
            if _norm(ref) in norm_idx:
                return norm_idx[_norm(ref)]
        close = difflib.get_close_matches(str(ref), names, n=1, cutoff=0.5)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(f"{who}: unknown event {ref!r}.{hint}")

    for s, st in enumerate(students):
        who = st.get("name", f"student #{s + 1}")
        student_names.append(who)
        officer[s] = bool(st.get("officer", False))

        perf = st.get("performance", {})
        if isinstance(perf, list):
            if len(perf) != E:
                raise ValueError(
                    f"{who}: performance list has {len(perf)} entries but "
                    f"there are {E} events (use null for events not taken).")
            for e, v in enumerate(perf):
                if v is not None:
                    raw[s, e] = float(v)
        elif isinstance(perf, dict):
            for key, v in perf.items():
                if v is None:
                    continue
                raw[s, resolve_event(key, who)] = float(v)
        else:
            raise ValueError(f"{who}: performance must be a list or an object.")

        # Preferences: 0-4 events, ranked best first
        prefs = st.get("preferences") or []
        if isinstance(prefs, (str, int)):
            prefs = [prefs]
        rank = 0
        seen = set()
        for ref in prefs:
            e = resolve_event(ref, who)
            if e in seen:
                warnings.append(f"{who} listed {names[e]} twice in preferences; "
                                f"kept the higher-ranked one.")
                continue
            if rank >= len(points):
                warnings.append(f"{who} has more than {len(points)} preferences; "
                                f"the extras were ignored.")
                break
            seen.add(e)
            pref[s, e] = points[rank]
            pref_rank[s, e] = rank + 1
            rank += 1

    return {
        "path": os.path.abspath(path),
        "ignored_settings": ignored_settings,
        "settings": settings,
        "event_names": names,
        "is_build": is_build,
        "types": types,
        "affinity": affinity,
        "time_blocks": time_blocks,
        "student_names": student_names,
        "raw": raw,
        "pref": pref,
        "pref_rank": pref_rank,
        "officer": officer,
        "warnings": warnings,
        "name_to_idx": name_to_idx,
        "lower_to_idx": lower_to_idx,
    }


# --------------------------------------------------------------------------
# Similarity matrix
# --------------------------------------------------------------------------
def fit_difficulty(raw, shrink=2.0, iters=100):
    """Additive model: score = overall + ability(student) - difficulty(test).

    Fitted on the observed scores only, so a test is not called "easy" just
    because strong students happened to take it (or "hard" because weaker
    students did). Returns (difficulty per event, ability per student).
    """
    taken = ~np.isnan(raw)
    X = np.where(taken, raw, 0.0)
    N, E = raw.shape
    if not taken.any():
        return np.zeros(E), np.zeros(N)
    mu = X[taken].mean()
    a = np.zeros(N)
    b = np.zeros(E)  # b = easiness; difficulty = -b
    n_e = taken.sum(axis=0)
    n_s = taken.sum(axis=1)
    for _ in range(iters):
        b = (np.where(taken, X - mu - a[:, None], 0).sum(axis=0)) / (n_e + shrink)
        a = (np.where(taken, X - mu - b[None, :], 0).sum(axis=1)) / (n_s + shrink)
        mu = (np.where(taken, X - a[:, None] - b[None, :], 0).sum()) / taken.sum()
    return -b, a


def zscore_by_event(raw, shrink=2.0):
    """Difficulty-adjusted, standardized scores.

    Each test's estimated difficulty is added back to its scores, so a 2/5 on
    a hard test and a 4/5 on an easy one are compared fairly. Then every test
    is put on a common spread. Unlike plain per-test z-scores, this does NOT
    subtract the average of whoever took the test, so a test taken mostly by
    strong students still shows those students as strong.
    """
    difficulty, _ = fit_difficulty(raw, shrink)
    taken = ~np.isnan(raw)
    adj = raw + difficulty[None, :]
    mu = np.nanmean(adj) if taken.any() else 0.0
    Z = np.full_like(raw, np.nan)
    for e in range(raw.shape[1]):
        m = taken[:, e]
        if m.sum() >= 2 and adj[m, e].std() > 1e-12:
            Z[m, e] = (adj[m, e] - mu) / adj[m, e].std()
        elif m.sum() >= 1:
            Z[m, e] = 0.0
    return Z


def cosine_matrix(V):
    norms = np.linalg.norm(V, axis=0)
    G = V.T @ V
    denom = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(denom > 0, G / denom, 0.0)
    np.fill_diagonal(C, 0.0)
    return C


def standardize_offdiag(M, mask=None):
    """z-score the off-diagonal entries (only those in mask, if given)."""
    E = M.shape[0]
    iu = np.triu_indices(E, 1)
    use = np.ones(len(iu[0]), dtype=bool) if mask is None else mask[iu]
    vals = M[iu][use]
    out = np.zeros_like(M)
    if len(vals) < 2 or vals.std() < 1e-12:
        return out
    Zm = (M - vals.mean()) / vals.std()
    if mask is not None:
        Zm = np.where(mask, Zm, 0.0)
    np.fill_diagonal(Zm, 0.0)
    return Zm


def performance_correlations(raw, settings):
    """Ability-adjusted residual correlation for every event pair."""
    Z = zscore_by_event(raw, settings["difficulty_shrinkage"])
    taken = ~np.isnan(raw)
    Zf = np.where(taken, Z, 0.0)
    zsum = Zf.sum(axis=1)
    cnt = taken.sum(axis=1)
    E = raw.shape[1]
    R = np.zeros((E, E))
    Nov = np.zeros((E, E), dtype=int)
    c = settings["ability_shrinkage"]
    for i, j in combinations(range(E), 2):
        m = taken[:, i] & taken[:, j]
        n = int(m.sum())
        Nov[i, j] = Nov[j, i] = n
        if n < max(3, settings["min_overlap"]):
            continue
        # General ability from the student's OTHER events (leave-pair-out)
        others_sum = zsum[m] - Zf[m, i] - Zf[m, j]
        others_cnt = cnt[m] - 2
        ability = others_sum / (others_cnt + c)
        ri = Zf[m, i] - ability
        rj = Zf[m, j] - ability
        if ri.std() < 1e-9 or rj.std() < 1e-9:
            continue
        r = float(np.corrcoef(ri, rj)[0, 1])
        R[i, j] = R[j, i] = r
    return R, Nov


def build_similarity(raw, pref, affinity, settings):
    E = raw.shape[1]
    taken = ~np.isnan(raw)

    R, Nov = performance_correlations(raw, settings)
    perf_ok = Nov >= settings["min_overlap"]
    np.fill_diagonal(perf_ok, False)
    Pz = standardize_offdiag(R, perf_ok)

    Prefz = standardize_offdiag(cosine_matrix(pref))

    takers = taken.sum(axis=0)
    has_tests = takers >= settings["min_overlap"]
    cosel_ok = np.outer(has_tests, has_tests)
    np.fill_diagonal(cosel_ok, False)
    Cz = standardize_offdiag(cosine_matrix(taken.astype(float)), cosel_ok)

    w_type = np.sqrt(np.outer(affinity, affinity))
    confidence = np.clip(Nov / settings["full_confidence_overlap"], 0, 1)
    w_perf = np.where(perf_ok, w_type * confidence, 0.0)

    beta = np.where(cosel_ok, settings["preference_share_of_rest"], 1.0)
    rest = beta * Prefz + (1 - beta) * Cz

    S = w_perf * Pz + (1 - w_perf) * rest
    np.fill_diagonal(S, 0.0)
    parts = {"R": R, "overlap": Nov, "perf_z": Pz, "pref_z": Prefz,
             "cosel_z": Cz, "w_perf": w_perf}
    return S, parts


# --------------------------------------------------------------------------
# Officer fit
# --------------------------------------------------------------------------
def officer_event_fit(raw, pref_rank, officer, affinity, settings):
    """Fit of each officer to each event, roughly on a -1..1 scale."""
    Z = zscore_by_event(raw, settings["difficulty_shrinkage"])
    taken = ~np.isnan(raw)
    takers = taken.sum(axis=0)
    off_idx = np.where(officer)[0]
    E = raw.shape[1]

    F = np.zeros((len(off_idx), E))
    for a, s in enumerate(off_idx):
        for e in range(E):
            w = affinity[e]
            if takers[e] < settings["min_overlap"]:
                w = 0.0  # no usable test for this event
            if taken[s, e]:
                perf_part = (float(np.clip(Z[s, e], -PERFORMANCE_Z_CLIP,
                                           PERFORMANCE_Z_CLIP)) / PERFORMANCE_Z_CLIP)
            else:
                perf_part = settings["untaken_event_fit"]
            r = int(pref_rank[s, e])
            pref_part = (PREFERENCE_FIT_BY_RANK[r - 1]
                         if 1 <= r <= len(PREFERENCE_FIT_BY_RANK) else 0.0)
            F[a, e] = w * perf_part + (1 - w) * pref_part
    return off_idx, F


# --------------------------------------------------------------------------
# The partition search
# --------------------------------------------------------------------------
BIG = 1000.0


def _officer_columns(G, settings, O):
    """Slot bonuses for the officer assignment problem."""
    pen = settings["officer_deviation_penalty"]
    T = settings["target_officers_per_group"]
    hard_min = settings["min_officers_per_group"] or 0
    if O < hard_min * G:
        hard_min = 0
    extra = OFFICER_OVERFLOW_SLOTS
    col_group, bonus = [], []
    for g in range(G):
        for k in range(T + extra):
            b = pen if k < T else -pen
            if k < hard_min:
                b += BIG
            col_group.append(g)
            bonus.append(b)
    return np.array(col_group), np.array(bonus), T + extra


def officer_value(fitG, cols, settings):
    """Best officer-to-group assignment for a fixed partition (exact).

    Each group has `target` slots worth a bonus (filling them removes the
    shortfall penalty) and two overflow slots that cost the penalty; an
    officer can also be left unassigned for 0. Solved as an assignment problem.
    """
    col_group, bonus, _ = cols
    O, G = fitG.shape
    lam = settings["officer_weight"]
    V = np.hstack([lam * fitG[:, col_group] + bonus, np.zeros((O, O))])
    r, c = linear_sum_assignment(V, maximize=True)
    value = V[r, c].sum()
    per_group = [[] for _ in range(G)]
    for o, col in zip(r, c):
        if col < len(col_group):
            per_group[col_group[col]].append(int(o))
    return value, per_group


def solve_partition(S, F, is_build, settings, hard_pairs, restarts, kicks, rng):
    """Iterated local search over swaps between groups (sizes stay exact)."""
    E = S.shape[0]
    sizes = np.array(settings["group_sizes"], dtype=float)
    G = len(sizes)
    Seff = S.copy()
    for i, j, kind in hard_pairs:
        v = -BIG if kind == "never" else BIG
        Seff[i, j] = Seff[j, i] = v
    maxb = settings.get("max_builds_per_group")
    bflag = is_build.astype(float)
    template = np.repeat(np.arange(G), settings["group_sizes"])
    swaps = list(combinations(range(E), 2))
    use_off = F is not None
    if use_off:
        cols = _officer_columns(G, settings, F.shape[0])
        lam = settings["officer_weight"]
        slots2 = 2 * cols[2]

    def build_pen(nb):
        return 0.0 if maxb is None else BIG * np.clip(nb - maxb, 0, None).sum()

    def local_search(assign):
        M = np.zeros((E, G))
        M[np.arange(E), assign] = 1
        GS = Seff @ M                 # event-to-group similarity sums
        nb = bflag @ M                # builds per group
        pair = 0.5 * float((GS * M).sum()) - build_pen(nb)
        if use_off:
            FS = F @ M                # officer fit sums per group
            ov, per = officer_value(FS / sizes, cols, settings)
        else:
            ov, per = 0.0, [[] for _ in range(G)]
        improved = True
        while improved:
            improved = False
            for idx in rng.permutation(len(swaps)):
                a, b = swaps[idx]
                A, B = assign[a], assign[b]
                if A == B:
                    continue
                dpair = (GS[a, B] + GS[b, A] - 2 * Seff[a, b]
                         - GS[a, A] - GS[b, B])
                if maxb is not None and bflag[a] != bflag[b]:
                    nb2 = nb.copy()
                    nb2[A] += bflag[b] - bflag[a]
                    nb2[B] += bflag[a] - bflag[b]
                    dpair += build_pen(nb) - build_pen(nb2)
                if use_off:
                    # Upper bound on how much the officer term could improve
                    diff = np.abs(F[:, a] - F[:, b])
                    k = min(slots2, len(diff))
                    ub = lam * np.sort(diff)[-k:].sum() / min(sizes[A], sizes[B])
                    if dpair + ub <= 1e-9:
                        continue
                    FS2 = FS.copy()
                    FS2[:, A] += F[:, b] - F[:, a]
                    FS2[:, B] += F[:, a] - F[:, b]
                    ov2, per2 = officer_value(FS2 / sizes, cols, settings)
                    delta = dpair + ov2 - ov
                else:
                    delta = dpair
                if delta > 1e-9:
                    assign[a], assign[b] = B, A
                    GS[:, A] += Seff[:, b] - Seff[:, a]
                    GS[:, B] += Seff[:, a] - Seff[:, b]
                    nb[A] += bflag[b] - bflag[a]
                    nb[B] += bflag[a] - bflag[b]
                    pair += dpair
                    if use_off:
                        FS, ov, per = FS2, ov2, per2
                    improved = True
        return pair + ov, per

    best_score, best_assign, best_per = -np.inf, None, None
    scores = []
    for _ in range(restarts):
        assign = rng.permutation(template)
        cur_sc, per = local_search(assign)
        cur = assign.copy()
        for _ in range(kicks):
            trial = cur.copy()
            for _ in range(2):  # random double swap, then re-optimize
                a, b = rng.choice(E, 2, replace=False)
                trial[a], trial[b] = trial[b], trial[a]
            tsc, tper = local_search(trial)
            if tsc > cur_sc + 1e-9:
                cur, cur_sc, per = trial, tsc, tper
        scores.append(cur_sc)
        if cur_sc > best_score + 1e-9:
            best_score, best_assign, best_per = cur_sc, cur.copy(), per

    hits = sum(abs(s_ - best_score) < 1e-6 for s_ in scores)
    groups = [sorted(np.where(best_assign == g)[0].tolist()) for g in range(G)]
    violated = []
    for i, j, kind in hard_pairs:
        together = best_assign[i] == best_assign[j]
        if (kind == "never" and together) or (kind == "must" and not together):
            violated.append((i, j, kind))
    return {"groups": groups, "officers": best_per, "objective": best_score,
            "hits": hits, "restarts": restarts, "violated": violated}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def assign_students(groups, Fall, ability, officer, names, settings, locks=None):
    """Place every student in exactly one event group (exact MILP).

    Maximizes how well students fit their group's events, while keeping head
    counts near their ideal (hard maximums), about 2 officers per group, and
    each group's mix of strength tiers close to the club-wide mix, leaning
    toward stronger members.
    """
    N = Fall.shape[0]
    G = len(groups)
    norm_special = {_norm(k): v for k, v in SPECIAL_GROUP_MAX.items()}

    ideal, cap, special = [], [], []
    for grp in groups:
        sp = [norm_special[_norm(names[e])] for e in grp
              if _norm(names[e]) in norm_special]
        mx = max(sp) if sp else MAX_PEOPLE_PER_GROUP
        cap.append(mx)
        special.append(bool(sp))
        ideal.append(min(IDEAL_PEOPLE_BY_EVENT_COUNT.get(len(grp),
                                                         MAX_PEOPLE_PER_GROUP), mx))
    if N > sum(cap):
        raise ValueError(f"{N} students but the group maximums only allow "
                         f"{sum(cap)}. Raise MAX_PEOPLE_PER_GROUP or "
                         f"SPECIAL_GROUP_MAX at the top of the script.")

    # Fit of each student to each group: mostly the average over the group's
    # events, plus credit for one standout event
    gfit = np.zeros((N, G))
    for g, grp in enumerate(groups):
        gfit[:, g] = (GROUP_FIT_AVERAGE_SHARE * Fall[:, grp].mean(axis=1)
                      + (1 - GROUP_FIT_AVERAGE_SHARE) * Fall[:, grp].max(axis=1))

    # Provisional teams from difficulty-adjusted ability. With
    # OFFICERS_OUTRANK_NON_OFFICERS, every officer is ranked ahead of every
    # non-officer, so no non-officer can be on a higher team than an officer.
    if OFFICERS_OUTRANK_NON_OFFICERS:
        order = np.lexsort((-ability, -officer.astype(int)))
    else:
        order = np.lexsort((-officer.astype(int), -ability))
    tier = np.full(N, len(TEAM_SIZES))
    start = 0
    for t, size in enumerate(TEAM_SIZES):
        tier[order[start:start + size]] = t
        start += size
    T = len(TEAM_SIZES) + 1
    tier_count = np.bincount(tier, minlength=T)

    # Expected size of each group, scaled so the totals match N
    exp = np.array(ideal, dtype=float) * N / sum(ideal)
    exp = np.minimum(exp, cap)
    exp *= N / exp.sum()

    nx = N * G
    xi = lambda s, g: s * G + g
    k = nx
    over = np.arange(k, k + G); k += G
    under = np.arange(k, k + G); k += G
    op = np.arange(k, k + G); k += G
    om = np.arange(k, k + G); k += G
    tp = np.arange(k, k + T * G).reshape(T, G); k += T * G
    tm = np.arange(k, k + T * G).reshape(T, G); k += T * G
    nv = k

    c = np.zeros(nv)
    for s_ in range(N):
        for g in range(G):
            c[xi(s_, g)] = -settings["student_fit_weight"] * gfit[s_, g]
    for g in range(G):
        c[over[g]] = (settings["oversize_penalty_special"] if special[g]
                      else settings["oversize_penalty"])
        c[under[g]] = settings["undersize_penalty"]
        c[op[g]] = c[om[g]] = settings["officer_balance_penalty"]
        for t in range(T):
            c[tp[t, g]] = c[tm[t, g]] = settings["tier_balance_penalty"]
        # Lean better: missing a top-tier member or having extra bottom-tier
        # members costs more than the reverse
        c[tm[0, g]] *= LEAN_BETTER_MULTIPLIER
        c[tp[T - 1, g]] *= LEAN_BETTER_MULTIPLIER

    rows, cols, vals, lb, ub = [], [], [], [], []

    def add(cs, vs, lo, hi):
        r = len(lb)
        rows.extend([r] * len(cs)); cols.extend(cs); vals.extend(vs)
        lb.append(lo); ub.append(hi)

    for s_ in range(N):
        add([xi(s_, g) for g in range(G)], [1] * G, 1, 1)
    off_idx = np.where(officer)[0]
    target_off = settings["target_officers_per_group"]
    for g in range(G):
        allx = [xi(s_, g) for s_ in range(N)]
        add(allx, [1] * N, 0, cap[g])                                    # hard max
        add(allx + [over[g], under[g]], [1] * N + [-1, 1], ideal[g], ideal[g])
        if len(off_idx):
            add([xi(s_, g) for s_ in off_idx] + [op[g], om[g]],
                [1] * len(off_idx) + [-1, 1], target_off, target_off)
        for t in range(T):
            mem = np.where(tier == t)[0]
            share = tier_count[t] * exp[g] / N
            add([xi(s_, g) for s_ in mem] + [tp[t, g], tm[t, g]],
                [1] * len(mem) + [-1, 1], share, share)
            if REQUIRE_EVERY_TEAM_IN_EVERY_GROUP and tier_count[t] >= G:
                add([xi(s_, g) for s_ in mem], [1] * len(mem), 1, np.inf)

    A = coo_matrix((vals, (rows, cols)), shape=(len(lb), nv)).tocsr()
    integrality = np.zeros(nv)
    integrality[:nx] = 1
    hi = np.full(nv, np.inf)
    hi[:nx] = 1
    # Locked students may only go to groups holding one of their lock events
    for s_, evs in (locks or {}).items():
        for g, grp in enumerate(groups):
            if not set(grp) & evs:
                hi[xi(s_, g)] = 0
    res = milp(c, constraints=LinearConstraint(A, lb, ub), integrality=integrality,
               bounds=Bounds(np.zeros(nv), hi),
               options={"time_limit": settings["assignment_time_limit_sec"],
                        "disp": False})
    if res.x is None:
        raise ValueError("Could not place every student within the group "
                         "limits" + (" and LOCKED_STUDENTS (too many people "
                                     "locked into one group, or locks that "
                                     "conflict with the team/officer rules?)"
                                     if locks else "") + ": " + str(res.message))
    X = res.x[:nx].reshape(N, G) > 0.5
    member_group = X.argmax(axis=1)
    return {"group_of": member_group, "tier": tier, "gfit": gfit,
            "ideal": ideal, "cap": cap, "optimal": res.status == 0}


def placement_metrics(group_of, ordered, raw, Zadj, pref_rank, points):
    """How well a placement matches preferences and performance.

    Works on one placement or many at once (group_of may be 2-D, one row per
    placement), so random baselines are cheap to score.
    """
    group_of = np.atleast_2d(group_of)
    E = raw.shape[1]
    EM = np.zeros((len(ordered), E), dtype=bool)
    for g, grp in enumerate(ordered):
        EM[g, grp] = True
    inG = EM[group_of]                                 # placements x N x E
    taken = ~np.isnan(raw)
    pts = np.zeros(pref_rank.shape)
    for r, v in enumerate(points, 1):
        pts[pref_rank == r] = v
    has_pref = pts.sum(axis=1) > 0
    pref_share = np.where(has_pref, (pts * inG).sum(axis=2)
                          / np.maximum(pts.sum(axis=1), 1e-9), np.nan)
    got_first = np.where(has_pref, ((pref_rank == 1) & inG).any(axis=2), np.nan)
    got_any = np.where(has_pref, ((pref_rank > 0) & inG).any(axis=2), np.nan)
    tk = taken & inG
    n_in = tk.sum(axis=2)
    Zf = np.where(taken, Zadj, 0.0)
    perf_in = np.where(n_in > 0, (Zf * tk).sum(axis=2) / np.maximum(n_in, 1), np.nan)
    n_all = taken.sum(axis=1)
    perf_all = np.where(n_all > 0, Zf.sum(axis=1) / np.maximum(n_all, 1), np.nan)
    perf_rel = perf_in - perf_all
    took_any = (n_in > 0).astype(float)
    with np.errstate(invalid="ignore"):
        summary = {
            "got_first_choice": np.nanmean(got_first, axis=1),
            "got_any_choice": np.nanmean(got_any, axis=1),
            "preference_points_captured": np.nanmean(pref_share, axis=1),
            "took_a_test_in_group": took_any.mean(axis=1),
            "avg_adjusted_score_on_group_tests": np.nanmean(perf_in, axis=1),
            "score_on_group_tests_vs_own_average": np.nanmean(perf_rel, axis=1),
        }
    per_student = {"pref_share": pref_share[0], "got_first": got_first[0],
                   "perf_in": perf_in[0], "perf_rel": perf_rel[0],
                   "took_any": took_any[0]}
    return summary, per_student


def resolve_locks(d):
    """Turn LOCKED_STUDENTS into {student index: set of event indices}."""
    snames = d["student_names"]
    by_norm = {}
    for i, n in enumerate(snames):
        by_norm.setdefault(_norm(n), []).append(i)
    ev_norm = {_norm(n): i for i, n in enumerate(d["event_names"])}
    locks = {}
    for who, evs in LOCKED_STUDENTS.items():
        hits = by_norm.get(_norm(who), [])
        if not hits:
            close = difflib.get_close_matches(who, snames, n=1, cutoff=0.6)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"LOCKED_STUDENTS: no student named {who!r} in the "
                             f"data file.{hint}")
        if len(hits) > 1:
            raise ValueError(f"LOCKED_STUDENTS: {who!r} matches more than one "
                             f"student; make the names in the data file unique.")
        if isinstance(evs, str):
            evs = [evs]
        idx = set()
        for ev in evs:
            if _norm(ev) not in ev_norm:
                close = difflib.get_close_matches(ev, d["event_names"], n=1, cutoff=0.5)
                hint = f" Did you mean {close[0]!r}?" if close else ""
                raise ValueError(f"LOCKED_STUDENTS: {who!r} is locked to unknown "
                                 f"event {ev!r}.{hint}")
            idx.add(ev_norm[_norm(ev)])
        if not idx:
            raise ValueError(f"LOCKED_STUDENTS: {who!r} has no events listed.")
        locks[hits[0]] = idx
    return locks


def resolve_pairs(d):
    st = d["settings"]
    out = []

    def idx(name):
        if name in d["name_to_idx"]:
            return d["name_to_idx"][name]
        if isinstance(name, str) and name.lower() in d["lower_to_idx"]:
            return d["lower_to_idx"][name.lower()]
        if isinstance(name, int):
            return name - st["preference_index_base"]
        raise ValueError(f"Unknown event in settings: {name!r}")

    # Cannot-link time blocks defined at the top of this script
    norm_to_idx = {_norm(n): i for i, n in enumerate(d["event_names"])}
    G = len(st["group_sizes"])
    for block, members in CANNOT_LINK_BLOCKS.items():
        ids = []
        for name in members:
            if _norm(name) not in norm_to_idx:
                close = difflib.get_close_matches(_norm(name), list(norm_to_idx),
                                                  n=1, cutoff=0.6)
                hint = (f"\nClosest event in the data file: "
                        f"{d['event_names'][norm_to_idx[close[0]]]!r}" if close else "")
                listing = "\n".join(f"  {ascii(n)}" for n in d["event_names"])
                raise ValueError(
                    f"Time block {block}: event {name!r} is not in the data file."
                    f"{hint}\nData file read: {d['path']}\n"
                    f"Events in that file (any odd characters show as \\u codes):\n"
                    f"{listing}\nFix the name either there or in CANNOT_LINK_BLOCKS "
                    f"at the top of scioly_groups.py.")
            ids.append(norm_to_idx[_norm(name)])
        if len(ids) > G:
            raise ValueError(f"Time block {block} has {len(ids)} events but there "
                             f"are only {G} groups, so they cannot all be split up.")
        for i, j in combinations(ids, 2):
            out.append((i, j, "never"))

    for a, b in st["never_together"]:
        out.append((idx(a), idx(b), "never"))
    for a, b in st["must_together"]:
        out.append((idx(a), idx(b), "must"))
    if st["avoid_same_time_block"]:
        tb = d["time_blocks"]
        for i, j in combinations(range(len(tb)), 2):
            if tb[i] is not None and tb[i] == tb[j]:
                out.append((i, j, "never"))
    return out


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "scioly_data.json")
    if not os.path.exists(path):
        print(f"Data file not found: {path}\n"
              f"Fill in scioly_template.json, save it as scioly_data.json next "
              f"to this script, or pass a path:  python scioly_groups.py my_data.json")
        sys.exit(1)

    d = load_data(path)
    st = d["settings"]
    names = d["event_names"]
    is_build = d["is_build"]
    types = d["types"]
    affinity = d["affinity"]
    mark = {"test": "", "build": " [B]", "hybrid": " [H]"}
    raw, pref, officer = d["raw"], d["pref"], d["officer"]
    N, E = raw.shape
    G = len(st["group_sizes"])
    rng = np.random.default_rng(st["random_seed"])

    print("=" * 72)
    print("SCIENCE OLYMPIAD EVENT GROUPING")
    print("=" * 72)
    takers = (~np.isnan(raw)).sum(axis=0)
    print(f"Students: {N}   Officers: {officer.sum()}   Events: {E} "
          f"({types.count('build')} build, {types.count('hybrid')} hybrid)   Groups: {G} sized {st['group_sizes']}")
    print(f"Average tests taken per student: {(~np.isnan(raw)).sum(1).mean():.1f}")

    warns = list(d["warnings"])
    if d["ignored_settings"]:
        warns.append("The data file's 'settings' block was ignored (edit the "
                     "CONFIGURATION section of the script instead). Differing "
                     "values: " + ", ".join(d["ignored_settings"]))
    for e in range(E):
        if types[e] != "build" and takers[e] < st["min_overlap"]:
            warns.append(f"Only {takers[e]} students took the {names[e]} test; "
                         f"it will be placed mostly by preference.")
    if officer.sum() < st["target_officers_per_group"] * G:
        warns.append(f"{officer.sum()} officers for a target of "
                     f"{st['target_officers_per_group']} x {G} groups; some groups "
                     f"will fall short.")
    if warns:
        print("\nNotes on the data:")
        for w in warns:
            print("  - " + w)

    hard_pairs = resolve_pairs(d)
    S, parts = build_similarity(raw, pref, affinity, st)
    off_idx, F = officer_event_fit(raw, d["pref_rank"], officer, affinity, st)

    difficulty, _ = fit_difficulty(raw, st["difficulty_shrinkage"])
    raw_means = np.array([np.nanmean(raw[:, e]) if takers[e] else np.nan
                          for e in range(E)])
    tested = [e for e in range(E) if takers[e] >= 2]
    if tested:
        print("\nTest difficulty (adjusted for who took each test; "
              "positive = harder than average):")
        for e in sorted(tested, key=lambda e: -difficulty[e]):
            print(f"  {difficulty[e]:+.2f}  {names[e]:<26} raw avg {raw_means[e]:.2f}, "
                  f"{takers[e]} takers")
    use_off = len(off_idx) > 0

    print(f"\nSearching for the best grouping ({st['search_restarts']} restarts)...")
    t0 = time.time()
    sol = solve_partition(S, F if use_off else None, is_build, st, hard_pairs,
                          st["search_restarts"], st["search_kicks"], rng)
    print(f"  done in {time.time() - t0:.1f}s; best grouping reached in "
          f"{sol['hits']} of {sol['restarts']} restarts")
    if sol["hits"] < 3:
        print("  (Few restarts agreed. Raise search_restarts in settings to be safe.)")
    for i, j, kind in sol["violated"]:
        print(f"  WARNING: could not satisfy {kind}_together for "
              f"{names[i]} + {names[j]}; the rules conflict with the group sizes.")

    # ---------------- Bootstrap stability ----------------
    B = int(st["bootstrap_runs"])
    co = np.zeros((E, E))
    ok_runs = 0
    if B > 0:
        print(f"Bootstrap stability: {B} resamples...")
        t0 = time.time()
        for b in range(B):
            samp = rng.integers(0, N, N)
            Sb, _ = build_similarity(raw[samp], pref[samp], affinity, st)
            sb = solve_partition(Sb, F if use_off else None, is_build, st,
                                 hard_pairs, st["bootstrap_restarts"],
                                 st["bootstrap_kicks"], rng)
            ok_runs += 1
            for grp in sb["groups"]:
                for i, j in combinations(grp, 2):
                    co[i, j] += 1
                    co[j, i] += 1
            print(f"\r  {b + 1}/{B}", end="", flush=True)
        print(f"\r  done in {time.time() - t0:.1f}s{' ' * 10}")
        if ok_runs:
            co /= ok_runs

    # ---------------- Report ----------------
    groups = sol["groups"]
    group_rows = []
    for g, grp in enumerate(groups):
        prs = list(combinations(grp, 2))
        cohesion = float(np.mean([S[i, j] for i, j in prs]))
        stab = float(np.mean([co[i, j] for i, j in prs])) if ok_runs else None
        group_rows.append((g, grp, cohesion, stab))
    group_rows.sort(key=lambda r: -r[2])

    print("\n" + "=" * 72)
    print("RECOMMENDED GROUPS   ([B] = build, [H] = hybrid build/test)")
    print("=" * 72)
    out_groups = []
    for rank, (g, grp, cohesion, stab) in enumerate(group_rows, 1):
        label = ", ".join(names[e] + mark[types[e]] for e in grp)
        stab_txt = f"   stability {stab:.0%}" if stab is not None else ""
        print(f"\nGroup {rank}: {label}")
        print(f"  cohesion {cohesion:+.2f}{stab_txt}")
        for i, j in combinations(grp, 2):
            n = parts["overlap"][i, j]
            r_txt = (f"perf r={parts['R'][i, j]:+.2f} (n={n})"
                     if n >= st["min_overlap"] else f"perf n/a (n={n})")
            co_txt = f", together {co[i, j]:.0%} of resamples" if ok_runs else ""
            print(f"    {names[i]} + {names[j]}: similarity {S[i, j]:+.2f}; "
                  f"{r_txt}; preference {parts['pref_z'][i, j]:+.2f}{co_txt}")
        if stab is not None:
            shaky = [names[e] for e in grp
                     if np.mean([co[e, k] for k in grp if k != e])
                     < UNCERTAIN_PLACEMENT_BELOW]
            if shaky:
                print(f"  Less certain placement: {', '.join(shaky)}")
        out_groups.append({
            "group": rank,
            "events": [names[e] for e in grp],
            "build_events": [names[e] for e in grp if types[e] == "build"],
            "hybrid_events": [names[e] for e in grp if types[e] == "hybrid"],
            "cohesion": round(cohesion, 3),
            "stability": None if stab is None else round(stab, 3),
        })

    iu = np.triu_indices(E, 1)
    top = sorted(zip(S[iu], iu[0], iu[1]), reverse=True)[:10]
    print("\nTop 10 most similar event pairs overall:")
    for v, i, j in top:
        print(f"  {v:+.2f}  {names[i]} + {names[j]}")

    # ---------------- Student placement ----------------
    ordered = [grp for _, grp, _, _ in group_rows]
    _, Fall = officer_event_fit(raw, d["pref_rank"], np.ones(N, dtype=bool),
                                affinity, st)
    _, ability = fit_difficulty(raw, st["difficulty_shrinkage"])
    locks = resolve_locks(d)
    asg = assign_students(ordered, Fall, ability, officer, names, st, locks)
    tier_names = "ABCDEFGH"
    T = len(TEAM_SIZES) + 1
    snames = d["student_names"]
    print("\n" + "=" * 72)
    print("STUDENT PLACEMENT   (* = officer; L = locked; A/B/C = provisional team)")
    print("=" * 72)
    roster_rows = []
    for g, grp in enumerate(ordered):
        mem = [s_ for s_ in range(N) if asg["group_of"][s_] == g]
        mem.sort(key=lambda s_: (asg["tier"][s_], -ability[s_]))
        tiers = np.bincount(asg["tier"][mem], minlength=T)
        mix = " ".join(f"{tier_names[t]}:{tiers[t]}" for t in range(T))
        n_off = int(officer[mem].sum())
        print(f"\nGroup {g + 1}: {', '.join(names[e] for e in grp)}")
        print(f"  {len(mem)} people (ideal {asg['ideal'][g]}, max {asg['cap'][g]})   "
              f"teams {mix}   officers {n_off}")
        out_groups[g]["members"] = []
        for s_ in mem:
            took = [f"{names[e]} {raw[s_, e]:g}" for e in grp if not np.isnan(raw[s_, e])]
            prefs = [f"#{int(d['pref_rank'][s_, e])} {names[e]}" for e in grp
                     if d["pref_rank"][s_, e] > 0]
            info = "; ".join(x for x in [", ".join(took), ", ".join(prefs)] if x)
            star = "*" if officer[s_] else " "
            lk = "L" if s_ in locks else " "
            tn = tier_names[asg["tier"][s_]]
            print(f"  {lk}{star}{tn}  {snames[s_]:<24} {info}")
            out_groups[g]["members"].append(
                {"name": snames[s_], "officer": bool(officer[s_]),
                 "locked": s_ in locks, "team": tn,
                 "group_fit": round(float(asg["gfit"][s_, g]), 3)})
            roster_rows.append([g + 1, snames[s_], "yes" if officer[s_] else "",
                                "yes" if s_ in locks else "", tn, f"{ability[s_]:.3f}",
                                f"{asg['gfit'][s_, g]:.3f}",
                                " | ".join(names[e] for e in grp)])
    if not asg["optimal"]:
        print("\n(Placement hit its time limit; result is the best found.)")

    # ---------------- Rule checks ----------------
    problems = []
    for g in range(G):
        mem = np.where(asg["group_of"] == g)[0]
        if len(mem) > asg["cap"][g]:
            problems.append(f"Group {g + 1} has {len(mem)} people (max {asg['cap'][g]}).")
        offs = [asg["tier"][x] for x in mem if officer[x]]
        nons = [asg["tier"][x] for x in mem if not officer[x]]
        if offs and nons and min(nons) < max(offs):
            problems.append(f"Group {g + 1}: a non-officer is on a higher team "
                            f"than an officer.")
    for i, j, kind in hard_pairs:
        same = any(i in grp and j in grp for grp in ordered)
        if (kind == "never" and same) or (kind == "must" and not same):
            problems.append(f"{names[i]} + {names[j]} break a {kind}-together rule.")
    for x, evs in locks.items():
        if not set(ordered[asg["group_of"][x]]) & evs:
            problems.append(f"{snames[x]} is not in a group with their locked event.")
    if len(np.unique(asg["group_of"])) < G:
        problems.append("At least one group has no students.")
    print("\nRule checks: " + ("all passed (sizes, officer team order, "
                               "time blocks, pair rules, locks)"
                               if not problems else ""))
    for p_ in problems:
        print("  PROBLEM: " + p_)

    # ---------------- Placement test ----------------
    Zadj = zscore_by_event(raw, st["difficulty_shrinkage"])
    points = st["preference_rank_points"]
    actual, per = placement_metrics(asg["group_of"], ordered, raw, Zadj,
                                    d["pref_rank"], points)
    nb = int(EVALUATION_RANDOM_BASELINES)
    if nb > 0:
        rand = np.array([rng.permutation(asg["group_of"]) for _ in range(nb)])
        base, _ = placement_metrics(rand, ordered, raw, Zadj, d["pref_rank"], points)
    labels = {
        "got_first_choice": ("Got their 1st choice in their group", True),
        "got_any_choice": ("Got at least one listed choice", True),
        "preference_points_captured": ("Preference points captured (rank-weighted)", True),
        "took_a_test_in_group": ("Took at least one test in their group", True),
        "avg_adjusted_score_on_group_tests": ("Avg difficulty-adjusted score on group tests (z)", False),
        "score_on_group_tests_vs_own_average": ("Score on group tests vs. own average (z)", False),
    }
    print("\n" + "=" * 72)
    print("PLACEMENT TEST" + (f"   (vs. {nb} random placements with the same "
                              f"group sizes)" if nb > 0 else ""))
    print("=" * 72)
    hdr = f"  {'':<50}{'this':>8}" + (f"{'random':>9}{'beats':>8}" if nb > 0 else "")
    print(hdr)
    for key, (label, pct) in labels.items():
        v = float(actual[key][0])
        fmt = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:+.2f}")
        line = f"  {label:<50}{fmt(v):>8}"
        if nb > 0:
            b = base[key]
            b = b[~np.isnan(b)]
            line += f"{fmt(float(b.mean())):>9}{(b < v).mean():>8.0%}"
        print(line)
    print("  (z: 0 = club average for that test; positive = better. 'beats' = share "
          "of random placements this one outscores.)")

    print(f"\n  {'Group':<7}{'people':>7}{'pref pts':>10}{'1st choice':>12}"
          f"{'took test':>11}{'score here':>12}{'avg ability':>13}")
    for g in range(G):
        mem = np.where(asg["group_of"] == g)[0]
        with np.errstate(invalid="ignore"):
            ps = np.nanmean(per["pref_share"][mem])
            gf = np.nanmean(per["got_first"][mem])
            pi = np.nanmean(per["perf_in"][mem])
        tk = per["took_any"][mem].mean()
        print(f"  {g + 1:<7}{len(mem):>7}{ps:>10.0%}{gf:>12.0%}{tk:>11.0%}"
              f"{pi:>+12.2f}{ability[mem].mean():>+13.2f}")
    print("  (avg ability: difficulty-adjusted points vs. club average; similar "
          "values mean balanced groups)")

    weak = [x for x in range(N) if per["took_any"][x] == 0
            and not (d["pref_rank"][x, ordered[asg["group_of"][x]]] > 0).any()]
    if weak:
        print("\n  Placed in a group with none of their choices and none of their "
              "tests (check by hand):")
        for x in weak:
            print(f"    {snames[x]} (group {asg['group_of'][x] + 1})")

    for row in roster_rows:
        x = snames.index(row[1])
        row += ["" if np.isnan(per["pref_share"][x]) else f"{per['pref_share'][x]:.2f}",
                "" if np.isnan(per["perf_in"][x]) else f"{per['perf_in'][x]:.2f}"]

    # ---------------- Save files ----------------
    out_dir = os.path.join(os.path.dirname(os.path.abspath(path)), "results")
    os.makedirs(out_dir, exist_ok=True)

    def write_matrix(fname, M, fmt="{:.3f}"):
        with open(os.path.join(out_dir, fname), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([""] + names)
            for i in range(E):
                w.writerow([names[i]] + [fmt.format(v) for v in M[i]])

    write_matrix("similarity_matrix.csv", S)
    write_matrix("performance_correlation.csv", parts["R"])
    write_matrix("test_overlap_counts.csv", parts["overlap"], "{:d}")
    with open(os.path.join(out_dir, "test_difficulty.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event", "type", "takers", "raw_average",
                    "difficulty (positive = harder)"])
        for e in range(E):
            w.writerow([names[e], types[e], int(takers[e]),
                        "" if np.isnan(raw_means[e]) else f"{raw_means[e]:.3f}",
                        f"{difficulty[e]:.3f}" if takers[e] >= 2 else ""])
    if ok_runs:
        write_matrix("bootstrap_together_rate.csv", co)
    if use_off:
        with open(os.path.join(out_dir, "officer_event_fit.csv"), "w",
                  newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["officer"] + names)
            for o, s in enumerate(off_idx):
                w.writerow([d["student_names"][s]] + [f"{v:.3f}" for v in F[o]])
    with open(os.path.join(out_dir, "student_assignments.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "student", "officer", "locked", "team", "ability",
                    "group_fit", "group_events", "preference_points_captured",
                    "avg_adjusted_score_on_group_tests"])
        w.writerows(roster_rows)
    with open(os.path.join(out_dir, "groups.json"), "w", encoding="utf-8") as f:
        json.dump({"groups": out_groups,
                   "search_restarts_reaching_best": f"{sol['hits']}/{sol['restarts']}",
                   "bootstrap_runs_completed": ok_runs,
                   "settings_used": st}, f, indent=2)
    print(f"\nSaved results to: {out_dir}")


if __name__ == "__main__":
    main()