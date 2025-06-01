# Dataset Information

## Available Datasets

### food_quality.csv
- 5 classes: Fresh, Slightly_Spoiled, Spoiled, Fermented, Contaminated
- 1000 samples
- 6 sensor readings per sample (MQ2, MQ3, MQ4, MQ5, MQ7, MQ135)

### How to collect your own data
1. Connect the E-Nose hardware
2. Run `python/data_collection.py`
3. Expose sensors to target odor
4. Label the samples appropriately

## Data Format
CSV files with columns:
- MQ2, MQ3, MQ4, MQ5, MQ7, MQ135: Sensor readings (0-1 normalized)
- Label: Classification label