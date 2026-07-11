# end-to-end-credit-card-fraud-detection-system
end-to-end-credit-card-fraud-detection-system

## Preprocessing Order

Run the notebooks in this order:

1. `notebooks/0_handle_missing_values.ipynb`
2. `notebooks/1_handle_outliers.ipynb`
3. `notebooks/2_feature_binning.ipynb`
4. `notebooks/handle_encoding.ipynb`
5. `notebooks/handle_scaling.ipynb`
6. `notebooks/2_handle_imbalance_smote.ipynb`

The feature-binning step should happen before encoding, scaling, and SMOTENC so the downstream notebooks receive the already-binned dataset and can treat those fields as categorical features.
