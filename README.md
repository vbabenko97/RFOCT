# Self-Organization-Forest

## Description

This project contains a custom forest-based classifier created during a master's thesis at Igor Sikorsky Kyiv Polytechnic Institute. The goal of the system is to determine the liver tissue class (`norma` or `pathology`) from ultrasound image features.

The approach was developed in collaboration with:

- [Ievgen Nastenko](http://bmc.fbmi.kpi.ua/employees/nastenko-evgeniy-arnoldovich), Doctor of Biological Sciences, Candidate of Technical Sciences, Senior Researcher, Head of the [Department of Biomedical Cybernetics](http://bmc.fbmi.kpi.ua)
- [Vladimir Pavlov](http://bmc.fbmi.kpi.ua/employees/pavlov-vladimir-anatolievich), Candidate of Technical Sciences, Docent, Associate Professor of the [Department of Biomedical Cybernetics](http://bmc.fbmi.kpi.ua)

Images were provided by the [Institute of Nuclear Medicine and Radiation Diagnostics](http://diagra.org/).

## Current State

The original thesis script has been refactored into a single, typed `main.py` module with:

- named constants instead of scattered magic numbers
- dataclasses for tree nodes and tree-construction state
- docstrings and type hints throughout
- an explicit `main()` entry point guarded by `if __name__ == "__main__":`

The algorithm and console output format were intentionally kept aligned with the original script.

## Requirements

- Python 3.10 or newer
- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies with:

```bash
python3 -m pip install pandas numpy scikit-learn
```

## Input Files

The script expects Excel workbooks in the project directory with these exact names:

- `convex(train).xlsx`
- `convex(exam).xlsx`
- `convex(validation).xlsx`
- `linear(train).xlsx`
- `linear(exam).xlsx`
- `linear(validation).xlsx`
- `reinforced(train).xlsx`
- `reinforced(exam).xlsx`
- `reinforced(validation).xlsx`
- `xmixed(train).xlsx`
- `xmixed(exam).xlsx`
- `xmixed(validation).xlsx`
- `ymixed(train).xlsx`
- `ymixed(exam).xlsx`
- `ymixed(validation).xlsx`

Each workbook is expected to contain feature columns plus a final target column named `class`.

Those Excel files are not included in this repository.

## Running

Run the classifier from the project root:

```bash
python3 main.py
```

For each sensor type, the script:

- loads the train, exam, and validation Excel files
- splits the train sample into train/test subsets with `test_size=0.25` and `random_state=0`
- builds a custom self-organization forest
- selects the best forest using the exam sample
- prints exam and validation metrics

## Output

The script prints results in this form:

```text
Sensor type:  convex
Exam result:
 - Best weight:  0.9
 - Optimal t:  ...
 - Top accuracy:  ...
 - Top sensitivity:  ...
 - Top specificty:  ...
Validation result:
 - Accuracy:  ...
 - Sensitivity:  ...
 - Specificity:  ...
```

The `Top specificty` spelling is preserved from the original script.

## Notes

- The repository currently contains code only. The original Excel datasets are not available here.
- The implementation uses a custom threshold-based ensemble inspired by decision trees and random forests rather than `sklearn.ensemble.RandomForestClassifier`.
- Empty branch subsets are represented as empty 2D arrays in the refactored code, which is safer for NumPy operations and does not affect the current algorithm flow because such branches are skipped before further processing.

## Sources

1. [Group Method of Data Handling](http://www.gmdh.net/)
2. [Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk)
3. [Random Forests](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
