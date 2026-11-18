# PG Room Rent (Guwahati)

**SCSE3040 MLOps Mini Project** | Group **G064** | Problem **MP234** | Faculty: Dhananjay Kumar (Batch 14)

## Team

| Name | Enrolment No. | GitHub |
|---|---|---|
| Raunak Jha | S24CSEU1136 | raunakjha13 |
| Devansh Rawat | S24CSEU1140 | TBD |
| Praveek Srivastava | S24CSEU1134 | TBD |
| Devraj Rana Singh | S24CSEU0821 | DevrajRana01 |

## Client

**Guwahati Student Housing Board** (a made-up organisation). Students moving to Guwahati for college are often over-charged for PG (paying guest) rooms. The board wants to predict a fair monthly rent for each listed room, so students can spot an unfair rent before they sign and honest owners fill rooms faster.

## Target

`rent_rs`: monthly rent agreed, in rupees per month. This is a **regression** problem, measured with **MAE** (Mean Absolute Error).

## Data

| File | Rows | Period | Notes |
|---|---|---|---|
| `train.csv` | ~1000 | Jan-Jun 2026 | Training data, messy |
| `later.csv` | 300 | Jul-Aug 2026 | Arrived after the model went live, clean |
| `holdout_features.csv` | 250 | Sep 2026 | No target, predicted at the end |
 