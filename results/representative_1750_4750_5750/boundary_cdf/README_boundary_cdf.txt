[boundary_only_cdf_analysis_x10]
- threshold mode: fixed
- thresholds: [-70.0, -65.0, -60.0]
- boundary width: ±10 dB around threshold
- error metric for CDF: boundary_mae_db
- boundary pixels: valid pixels satisfying |GT_RSS - threshold| <= boundary_width_dbm
- boundary-only error CDF: CDF of sample-wise boundary MAE/RMSE
- RSS error CDF x-axis is set to 0-10 dB by default
- boundary-only Miss Rate CDF: CDF of FN / GT-shadow pixels within boundary pixels