[final_eval_metrics_fixed_threshold_heatmap]
- Shadow labels are defined using fixed absolute RSS thresholds.
- thresholds: [-70.0, -65.0, -60.0]
- pred_shadow = pred_RSS < threshold
- gt_shadow   = gt_RSS   < threshold
- building / excluded pixels are removed from evaluation.
- heatmaps use sample-wise mean Precision / Recall / IoU / F1-score.
- summary CSV also includes micro metrics from accumulated TP/FP/FN/TN.
- RMSE and MAE are threshold-independent but saved for consistency.