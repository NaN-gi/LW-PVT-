# Data folder

The ASHRAE RP-1043 chiller data are licensed by ASHRAE and are **not** redistributed with this code.
The data set "RP-1043: Fault Detection and Diagnostic Requirements and Evaluation Tools for Chillers"
is available from the ASHRAE store.

Place the original test files here:

```
data/raw/ASHRAE_RP-1043/Chiller Data/read me first.xls
data/raw/ASHRAE_RP-1043/Chiller Data/<fault folder>/<test file>.xls      (all folders of the data set)
```

Then run, from the repository root, the five scripts of `prepare/` in order. They write

```
data/processed/rp1043_steady.csv                          operating points of all test files
data/processed/rp1043_complete.parquet                    10-second time series of all test files
data/processed/rp1043_steady_normal.csv                   486 normal-operation operating points
data/processed/splits/splits_steady_normal.parquet        role of each operating point in the 15 scenarios
data/processed/gathered_normal/steady_normal_gathered01.csv   35,641 data within +/-1% of an operating point
data/processed/splits/splits_gathered_normal_1.parquet    role of each of the 35,641 data in the 15 scenarios
```

The training scripts read the last two files. Both folders are ignored by git.
