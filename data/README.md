### Data folders

There are four subfolders with data:
- './input_raw/':
    - In this folder, the original, raw (private!!) data is saved.
    It holds both the original raw, and the preprocessed data.
    The content of this folder is confidential, an cannot be shared!
- './input_agg/':
    - In this folder, the aggregated input data is saved.
    This data is used as the input to the simulator, and can be shared after permission from the data owner.
- './output_raw/':
    - In this folder, the raw output data from the simulator is saved.
    It has the same structure as the (preprocessed) raw input data.
- './output_agg/':
    - In this folder, the aggregated output data is saved.

### Processing of raw input data

If you have the original data ('./input_raw/anonymized_dataset.csv'), run in the following order:

- preprocess_data.py
    -> generates './input_raw/anonymized_dataset_preprocessed.csv'
- analyse_data.ipynb (optional)
    -> iPython notebook with some nice graphs
- aggregate_data.py
    -> generates the data which is used for the input of the simulator and saves it in './input_agg/'

### Comparison of aggregated input and output data

If you have an output file from the simulator ('./output_raw/dataset.csv'), run the following:

- compare_data