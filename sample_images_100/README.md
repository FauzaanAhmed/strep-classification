# Strep Throat Classification Dataset

## Overview

This dataset contains **100 throat images** for developing a binary classifier to distinguish between strep-positive and strep-negative cases. The dataset is balanced with 50 positive and 50 negative samples.

## Dataset Contents

### Images
- **100 throat images** in JPEG format
- Images are cropped views of the throat/pharynx region


### Labels & Symptoms (`sample_dataset_100.csv`)

| Column | Description |
|--------|-------------|
| `ImageName` | Filename of the corresponding throat image |
| `label` | Ground truth classification: `Positive` (strep-positive) or `Negative` (strep-negative) |
| `Hoarseness` | Patient reports hoarse voice (0 = No, 1 = Yes) |
| `Rhinorrhea` | Runny nose present (0 = No, 1 = Yes) |
| `sorethroat` | Patient reports sore throat (0 = No, 1 = Yes) |
| `Congestion` | Nasal congestion present (0 = No, 1 = Yes) |
| `Knownrecentcontact` | Known recent contact with strep-positive individual (0 = No, 1 = Yes) |
| `Headache` | Patient reports headache (0 = No, 1 = Yes) |
| `Fever` | Patient has fever (0 = No, 1 = Yes) |

## Class Distribution

- **Positive (strep-positive):** 50 images
- **Negative (strep-negative):** 50 images

## Task

Build a deep learning model to classify throat images as strep-positive or strep-negative. You may optionally incorporate the clinical symptom features to improve classification performance.

## Notes

- All symptom values are binary (0 or 1)
- Images vary in lighting and positioning but are all focused on the throat region
- This is a subset sampled from a larger clinical dataset
