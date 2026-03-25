# Data Dictionary

## Main Dataset: `data/interim/merged_filtered.csv`

| Column | Type | Description | Role |
|---|---|---|---|
| `objid` | integer/string identifier | SDSS object identifier for each galaxy | Traceability key |
| `gz2_class` | categorical string | Morphology class label from Hart (2016) / Galaxy Zoo 2 mapping | Target variable |
| `image_path` | string path | Absolute or local filesystem path to the `.jpg` image | Model input locator |

## Upstream Sources

| Source | Format | Description |
|---|---|---|
| `gz2_filename_mapping.csv` | CSV | Maps `objid` to image `asset_id` |
| `gz2_hart16.csv.gz` | compressed CSV | Contains morphology labels, including `dr7objid` and `gz2_class` |

## Derived Fields During Acquisition

| Field | Description |
|---|---|
| `objid` | Standardized key after renaming `dr7objid` |
| `image_path` | Built from image root directory + `asset_id` + `.jpg` |

## Notes

- The repository keeps only lightweight metadata artifacts under version control.
- Large raw images and heavy intermediate artifacts are local-only.
