# Science Olympiad Event Grouping & Student Placement

A Python tool that groups Science Olympiad events into study groups and places every club member into one of those groups. It uses placement-test scores, ranked event preferences, competition time blocks, and officer roles.

Events whose results tend to move together are grouped together. Build events lean on preference rather than test scores. Each group gets a balanced, slightly top-leaning mix of members, anchored by officers.

## Features

- **Fair scoring across tests.** Estimates each test's difficulty and each student's ability together, so a hard test does not penalize the students who took it.
- **Correlation-based event grouping.** Pairs events by how performance on one predicts performance on the other, after removing general ability.
- **Preference-aware.** Uses ranked preferences (0–4 events per student, best first). Build events weigh preference above performance.
- **Hard scheduling rules.** Events in the same competition time block are never grouped together. Optional must-pair and never-pair rules are also supported.
- **Complete student placement.** Every student is placed in exactly one group, with:
  - size limits, including a larger limit for the Codebusters group;
  - about 2 officers per group;
  - a balanced mix of provisional A/B/C teams.
- **Officer ordering.** Officers are never placed on a lower provisional team than a non-officer in their group.
- **Locks.** Specific students can be forced into a group containing a chosen event.
- **Built-in validation.** A bootstrap stability check, independent rule checks, and a placement test against thousands of random placements.

## Quick start

### Requirements

- Python 3.9+
- NumPy and SciPy (SciPy 1.9 or newer)

```bash
pip install numpy scipy
```

### Run

1. Fill in your data using `scioly_template.json` as a guide, and save it as `scioly_data.json` in the same folder as the script.
2. Run:

```bash
python scioly_groups.py                # reads scioly_data.json
python scioly_groups.py my_data.json   # or any other data file
```

A full run takes about a minute. Results print to the terminal and are saved to a `results/` folder next to the data file.

## Data format

```json
{
  "events": [
    {"name": "Boomilever", "type": "build"},
    {"name": "Thermodynamics", "type": "hybrid"},
    {"name": "Anatomy and Physiology", "type": "test"}
  ],
  "students": [
    {
      "name": "Jane Doe",
      "officer": true,
      "performance": {"Anatomy and Physiology": 4, "Botany": 5},
      "preferences": ["Anatomy and Physiology", "Botany", "Wright Stuff"]
    }
  ]
}
```

| Field | Description |
|---|---|
| `events[].type` | `test`, `build`, or `hybrid` (a build with a test component, e.g. Hovercraft) |
| `performance` | Only the tests the student took, as `"Event Name": score`. Use raw scores; difficulty is corrected automatically |
| `preferences` | 0–4 event names, best first. Event numbers also work (1 = first event listed) |
| `officer` | `true` for officers |

Event names ignore case, spacing and punctuation. A misspelled name stops the run and suggests the closest match.

## Configuration

All rules and weights are variables in the **`CONFIGURATION`** section at the top of `scioly_groups.py`. The data file supplies only events and students. A `settings` block in the JSON is ignored unless `USE_JSON_SETTINGS = True`.

### Common settings

```python
GROUP_SIZES = [3, 3, 3, 3, 3, 4, 4]        # events per group

CANNOT_LINK_BLOCKS = {                      # same-time tests; never grouped together
    "A": ["Anatomy and Physiology", "Astronomy", "Chemistry Lab", "Codebusters"],
    "B": ["Botany", "Dynamic Planet", "Forensics"],
    "C": ["Designer Genes", "Remote Sensing", "Write-it/Do-It"],
    "D": ["Protein Modeling", "Rocks and Minerals", "Water Quality"],
}

MAX_PEOPLE_PER_GROUP = 7
IDEAL_PEOPLE_BY_EVENT_COUNT = {3: 6, 4: 7}
SPECIAL_GROUP_MAX = {"Codebusters": 9}      # Codebusters allows 3 per team

TEAM_SIZES = [15, 15]                       # provisional A and B teams; rest = C
TARGET_OFFICERS_PER_GROUP = 2
```

### Locking students to events

```python
LOCKED_STUDENTS = {
    "Jane Doe":   ["Codebusters"],
    "John Smith": ["Circuit Lab", "Thermodynamics"],   # either one satisfies the lock
}
```

These lines go **inside the script file**, not the terminal. If locks cannot all be satisfied within the size and team rules, the run stops and says so.

See [Tunable variables](#tunable-variables) for the full list.

## Output

### Terminal

- Test difficulty ranking
- Recommended event groups with cohesion, stability, and the evidence behind each pairing
- Student placement for each group: head count, team mix, officers, and each member's scores and preferences within the group
- Rule checks
- Placement test comparing the result with random placements

### `results/` folder

| File | Contents |
|---|---|
| `groups.json` | Groups, members, and the settings used |
| `student_assignments.csv` | One row per student: group, officer, locked, team, ability, fit, preference capture, score on group tests |
| `similarity_matrix.csv` | Final event-to-event similarity |
| `performance_correlation.csv` | Ability-adjusted performance correlations |
| `test_overlap_counts.csv` | Number of students who took each pair of tests |
| `test_difficulty.csv` | Estimated difficulty of each test |
| `bootstrap_together_rate.csv` | How often each pair of events was grouped together across resamples |
| `officer_event_fit.csv` | Each officer's fit to each event |

## How it works

The model is a seven-stage pipeline.

```
1. Difficulty correction
2. Event similarity      ┐
3. Person–event fit      ┘─► 4. Event grouping ─► 5. Bootstrap stability
                                    │
                                    ▼
                          6. Student placement ─► 7. Checks & placement test
```

### 1. Difficulty correction

Scores are modeled as:

```
score = overall average + student ability − test difficulty + noise
```

The model is fitted with shrinkage on observed scores only. Because ability and difficulty are estimated together, a test is not called easy just because strong students chose it. Scores are then converted to difficulty-adjusted z-scores, and each student's fitted ability is reused as their overall strength.

### 2. Event-to-event similarity

Three signals are computed for every pair of events, standardized, and blended:

- **Performance correlation.** Among students who took both tests, the correlation of their scores after subtracting general ability. Ability is estimated from each student's *other* tests, so it does not absorb the signal being measured.
- **Preference affinity.** Cosine similarity of rank-weighted preferences (4/3/2/1 points). It is high when the same students rank both events highly.
- **Co-selection affinity.** Cosine similarity of which tests students chose to take.

```
similarity = w × performance + (1 − w) × [ β × preference + (1 − β) × co-selection ]
w = √(trust of event 1 × trust of event 2) × min(1, shared test-takers ÷ FULL_CONFIDENCE_OVERLAP)
```

Trust is highest for tests, intermediate for hybrids, and lowest for builds. Pairs with few shared test-takers automatically lean on preference.

### 3. Person-to-event fit

```
fit = trust × performance part + (1 − trust) × preference part
```

- The performance part is the capped adjusted z-score, or a small penalty for skipped tests.
- The preference part comes from the rank (1.0 / 0.8 / 0.6 / 0.4), so earlier choices always count more.

### 4. Event grouping

This is a capacitated graph-partitioning problem. It maximizes the total similarity of event pairs that share a group, plus an officer-coverage value. That value is solved exactly as an assignment problem (Hungarian algorithm) for each candidate grouping.

Time blocks and pair rules are hard constraints. The search is an iterated local search over event swaps with many random restarts. The output reports how many restarts reached the same best answer, which is practical evidence of a global optimum.

### 5. Bootstrap stability

Students are resampled with replacement, the similarity matrix is rebuilt, and the grouping is re-solved about 100 times. The share of runs in which two events stay together measures how much the result depends on chance. Weak pairings are flagged for human judgment.

### 6. Student placement

This is a mixed-integer linear program solved exactly with HiGHS (bundled with SciPy).

- **Provisional teams.** Students are ranked by ability, with officers first, and cut into teams A/B/C.
- **Objective.** Maximizes total student–group fit, where group fit is 60% average fit over the group's events plus 40% best single event.
- **Hard constraints.**
  - Each student is in exactly one group.
  - Group maximums are respected.
  - Every team is represented in every group.
  - Locks are honored.
- **Soft targets.**
  - Group sizes stay near their ideal.
  - About 2 officers per group.
  - Each group's team mix matches the club-wide mix, with extra weight on keeping top-team members present and bottom-team surplus low.

**Codebusters note:** Codebusters is the only 3-person event; all others allow 2 per team. The Codebusters group therefore holds 3 people per team, who rotate through the 2-person events. If Codebusters is in a 4-event group, each person does Codebusters plus 2 of the other 3 events, and every event is exactly filled.

### 7. Checks and placement test

The final placement is independently checked for sizes, officer team order, time blocks, pair rules, and locks. It is then scored on six metrics:

- 1st choice received
- any choice received
- rank-weighted preference points captured
- took a test in their group
- adjusted score on group tests
- score on group tests vs. the student's own average

Each metric is compared with 2,000 random placements that have identical group sizes (a permutation test).

## Tunable variables

| Area | Variable | Default | Effect of increasing / changing |
|---|---|---|---|
| Settings source | `USE_JSON_SETTINGS` | `False` | `True` lets a JSON `settings` block override the script |
| Event groups | `GROUP_SIZES` | `[3,3,3,3,3,4,4]` | Number and size of event groups (must sum to the event count) |
| | `CANNOT_LINK_BLOCKS` | blocks A–D | Events in one block are never grouped together |
| | `NEVER_TOGETHER`, `MUST_TOGETHER` | empty | Extra hard pair rules |
| | `MAX_BUILDS_PER_GROUP` | `None` | Caps build/hybrid events per group |
| | `AVOID_SAME_TIME_BLOCK` | `False` | Also applies `time_block` fields from the data file |
| Preferences | `PREFERENCE_RANK_POINTS` | `[4,3,2,1]` | Steeper values make 1st choices dominate event similarity |
| | `PREFERENCE_FIT_BY_RANK` | `[1.0,0.8,0.6,0.4]` | Steeper values make 1st choices dominate placement |
| | `PREFERENCE_INDEX_BASE` | `1` | Numbering used when preferences are numbers |
| Performance vs. preference | `PERF_WEIGHT_TEST` / `_HYBRID` / `_BUILD` | 0.65 / 0.40 / 0.15 | More reliance on test results for that event type |
| | `PREFERENCE_SHARE_OF_REST` | 0.75 | More weight on rankings vs. which tests were taken |
| | `DIFFICULTY_SHRINKAGE` | 2.0 | More cautious difficulty estimates |
| | `ABILITY_SHRINKAGE` | 1.0 | More cautious ability removal in correlations |
| | `MIN_OVERLAP` | 4 | More shared test-takers required before a correlation is used |
| | `FULL_CONFIDENCE_OVERLAP` | 12 | More shared test-takers required for full performance weight |
| | `UNTAKEN_EVENT_FIT` | −0.25 | Less penalty for skipped tests (toward 0) |
| | `PERFORMANCE_Z_CLIP` | 2.0 | Extreme scores count more |
| Officers | `TARGET_OFFICERS_PER_GROUP` | 2 | Officers sought per group |
| | `MIN_OFFICERS_PER_GROUP` | 1 | Hard floor when grouping events |
| | `OFFICER_WEIGHT` | 1.0 | Officer fit shapes event groups more strongly |
| | `OFFICER_DEVIATION_PENALTY` | 2.0 | Stricter officer count when grouping events |
| | `OFFICER_OVERFLOW_SLOTS` | 2 | More officers allowed above target in one group |
| Placement | `MAX_PEOPLE_PER_GROUP` | 7 | Hard size limit |
| | `IDEAL_PEOPLE_BY_EVENT_COUNT` | `{3: 6, 4: 7}` | Target size by number of events |
| | `SPECIAL_GROUP_MAX` | `{"Codebusters": 9}` | Larger limit for groups containing these events |
| | `TEAM_SIZES` | `[15, 15]` | Sizes of provisional teams A and B (rest = C) |
| | `OFFICERS_OUTRANK_NON_OFFICERS` | `True` | Officers fill the highest teams first |
| | `REQUIRE_EVERY_TEAM_IN_EVERY_GROUP` | `True` | Every group gets at least one A, B and C member |
| | `GROUP_FIT_AVERAGE_SHARE` | 0.6 | Rewards broad fit over one standout event |
| | `STUDENT_FIT_WEIGHT` | 1.0 | Fit matters more relative to balance rules |
| | `OVERSIZE_PENALTY` / `_SPECIAL` | 1.5 / 0.25 | Stronger resistance to oversized groups |
| | `UNDERSIZE_PENALTY` | 1.0 | Stronger resistance to undersized groups |
| | `OFFICER_BALANCE_PENALTY` | 0.75 | Stricter officer count per group |
| | `TEAM_BALANCE_PENALTY` | 0.5 | Stricter talent balance across groups |
| | `LEAN_BETTER_MULTIPLIER` | 1.5 | Stronger push toward top-heavy groups |
| | `LOCKED_STUDENTS` | empty | Forces listed students into a group with one of their listed events |
| | `ASSIGNMENT_TIME_LIMIT_SEC` | 60 | More time for the placement solver |
| Search & evaluation | `SEARCH_RESTARTS`, `SEARCH_KICKS` | 100, 10 | More thorough event-group search (slower) |
| | `BOOTSTRAP_RUNS` | 100 | More precise stability estimates (0 to skip) |
| | `BOOTSTRAP_RESTARTS`, `BOOTSTRAP_KICKS` | 10, 5 | More thorough search per resample |
| | `UNCERTAIN_PLACEMENT_BELOW` | 0.4 | Flags more events as uncertain |
| | `EVALUATION_RANDOM_BASELINES` | 2000 | More precise random comparison |
| | `RANDOM_SEED` | 42 | Different random draws; results should be broadly similar |

**Tuning tip:** change one variable at a time and compare three things:
- the restart agreement count;
- group stability;
- the placement-test table.

Gains in one area often cost another, so judge changes against the club's priorities.

## Limitations

- **Small samples.** Five questions per test and few shared test-takers per pair make correlations noisy. Shrinkage, confidence weighting and the bootstrap reduce this problem and expose it, but cannot remove it.
- **Self-selection.** Students chose their tests, so missing scores are not random. The joint difficulty model and co-selection signal partly account for this.
- **Additive difficulty.** The model assumes a test shifts everyone's score equally. Floor and ceiling effects on a 0–5 scale can violate this.
- **Thin ability estimates.** Students who took only one or two tests have imprecise ability estimates.
- **Judgment-based weights.** Weights encode priorities, not facts. Different reasonable settings give different, equally defensible results.
- **Provisional teams.** The A/B/C labels balance talent across groups. They are not final competition rosters.

## Project files

| File | Purpose |
|---|---|
| `scioly_groups.py` | The model and all configuration |
| `scioly_template.json` | Data file template |
| `sample_data.json` | Example data for testing (fictional students) |