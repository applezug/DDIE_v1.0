# Datasets (not included)

This repository does **not** include any datasets.

Please download datasets yourself and place them under this folder using the expected structure:

```
Data/datasets/
├── nasa_battery/
│   ├── B0005.mat
│   ├── B0006.mat
│   ├── B0007.mat
│   └── B0018.mat
└── NASA IGBT/
    └── Data/
        └── SMU Data for new devices/
            ├── IGBT-IRG4BC30K/
            │   ├── Part 1/LeakageIV.csv
            │   ├── Part 2/LeakageIV.csv
            │   └── ...
            └── MOSFET-IRF520Npbf/
                ├── Part 1/LeakageIV.csv
                └── ...
```

Notes:

- File names / exact subset should follow the paper.
- If dataset files are missing, some scripts may fall back to **synthetic data** for sanity checks (not paper numbers).

## Aligning dataset locations with configs (`data_root`)

If you place datasets outside this repository, update the corresponding `Config/*.yaml` field `dataloader.*.params.data_root`.

Examples:

```yaml
# Example: NASA battery extracted outside the repo
data_root: "D:/data/nasa_battery/NASA Battery Data Set/5. Battery Data Set"
```

```yaml
# Example: NASA IGBT extracted outside the repo
data_root: "D:/data/NASA_IGBT/Data/SMU Data for new devices"
```

If you keep data inside the repo, use relative paths (default configs already do so), e.g.:

```yaml
data_root: "./Data/datasets/nasa_battery/NASA Battery Data Set/5. Battery Data Set"
data_root: "./Data/datasets/NASA IGBT/Data/SMU Data for new devices"
```

