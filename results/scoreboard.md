# Bearing benchmark scoreboard

CWRU drive-end bearing, 40 recordings (4 healthy, 12 inner race, 12 outer race, 12 ball faults), scored on the same 1-second centre segment of each recording.

| Contender | Accuracy | Correct | Normal | Inner race | Outer race | Ball | Setup |
|---|---|---|---|---|---|---|---|
| Classic ML, random split | 100% | 119/119 |  |  |  |  | 1-s pieces of the same recording in train and test (leaky) |
| Classic ML, unseen motor load | 100% | 40/40 | 4/4 | 12/12 | 12/12 | 12/12 | same damaged bearing seen in training at other loads |
| Classic ML, unseen bearing | 80% | 32/40 | 4/4 | 9/12 | 8/12 | 11/12 | test bearing never seen in training |
| Claude, raw numbers | 82% | 33/40 | 4/4 | 10/12 | 7/12 | 12/12 | 12,000 samples pasted in the prompt, no tools, no training |
| Claude + vibtool report | 82% | 33/40 | 4/4 | 12/12 | 8/12 | 9/12 | one-page envelope analysis report, no tools, no training |

## Claude runs

| Contender | Model | Avg prompt tokens | Confidence when right | Confidence when wrong | Unparsed | Cost (USD, API list price) |
|---|---|---|---|---|---|---|
| Claude, raw numbers | claude-opus-5-5 | 66,901 | 0.67 | 0.51 | 0 | 22.30 |
| Claude + vibtool report | claude-opus-5-5 | 2,392 | 0.74 | 0.54 | 0 | 1.10 |

## Every recording

Misses in bold.

| Recording | Truth | Classic ML, unseen motor load | Classic ML, unseen bearing | Claude, raw numbers | Claude + vibtool report |
|---|---|---|---|---|---|
| Normal_0 | normal | normal | normal | normal | normal |
| Normal_1 | normal | normal | normal | normal | normal |
| Normal_2 | normal | normal | normal | normal | normal |
| Normal_3 | normal | normal | normal | normal | normal |
| IR007_0 | inner_race | inner_race | inner_race | inner_race | inner_race |
| IR007_1 | inner_race | inner_race | inner_race | inner_race | inner_race |
| IR007_2 | inner_race | inner_race | inner_race | inner_race | inner_race |
| IR007_3 | inner_race | inner_race | inner_race | inner_race | inner_race |
| B007_0 | ball | ball | ball | ball | ball |
| B007_1 | ball | ball | ball | ball | ball |
| B007_2 | ball | ball | ball | ball | ball |
| B007_3 | ball | ball | ball | ball | ball |
| OR007@6_0 | outer_race | outer_race | outer_race | outer_race | outer_race |
| OR007@6_1 | outer_race | outer_race | outer_race | outer_race | outer_race |
| OR007@6_2 | outer_race | outer_race | outer_race | outer_race | outer_race |
| OR007@6_3 | outer_race | outer_race | outer_race | outer_race | outer_race |
| IR014_0 | inner_race | inner_race | inner_race | **ball** | inner_race |
| IR014_1 | inner_race | inner_race | **ball** | inner_race | inner_race |
| IR014_2 | inner_race | inner_race | **ball** | inner_race | inner_race |
| IR014_3 | inner_race | inner_race | **ball** | **ball** | inner_race |
| B014_0 | ball | ball | ball | ball | ball |
| B014_1 | ball | ball | ball | ball | **inner_race** |
| B014_2 | ball | ball | ball | ball | ball |
| B014_3 | ball | ball | ball | ball | **inner_race** |
| OR014@6_0 | outer_race | outer_race | **ball** | **ball** | **ball** |
| OR014@6_1 | outer_race | outer_race | **ball** | **ball** | **inner_race** |
| OR014@6_2 | outer_race | outer_race | **ball** | **ball** | **ball** |
| OR014@6_3 | outer_race | outer_race | **ball** | **ball** | **inner_race** |
| IR021_0 | inner_race | inner_race | inner_race | inner_race | inner_race |
| IR021_1 | inner_race | inner_race | inner_race | inner_race | inner_race |
| IR021_2 | inner_race | inner_race | inner_race | inner_race | inner_race |
| IR021_3 | inner_race | inner_race | inner_race | inner_race | inner_race |
| B021_0 | ball | ball | ball | ball | ball |
| B021_1 | ball | ball | ball | ball | ball |
| B021_2 | ball | ball | **outer_race** | ball | **inner_race** |
| B021_3 | ball | ball | ball | ball | ball |
| OR021@6_0 | outer_race | outer_race | outer_race | outer_race | outer_race |
| OR021@6_1 | outer_race | outer_race | outer_race | **inner_race** | outer_race |
| OR021@6_2 | outer_race | outer_race | outer_race | outer_race | outer_race |
| OR021@6_3 | outer_race | outer_race | outer_race | outer_race | outer_race |
