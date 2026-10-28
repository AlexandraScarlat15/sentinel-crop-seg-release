"""U-Net-only public evaluation namespace. Generated for curated release.

Development repository retains optional Random Forest compatibility,
but the distributed package deliberately contains no RF modules.
"""
from sentinel_crop_seg.core.evaluation.checkpoint import (
    LoadedUNetCheckpoint, UNetCheckpointConfig, build_unet_from_config,
    extract_unet_checkpoint_config, load_unet_checkpoint, safe_torch_load,
)
from sentinel_crop_seg.core.evaluation.evaluator import (
    EvaluationResult, SegmentationPredictor, evaluate_segmentation_predictor,
)
from sentinel_crop_seg.core.evaluation.metadata_filters import (
    month_is_allowed, normalize_sample_id, parse_scene_month,
    read_allowed_manifest_sample_keys, read_allowed_sample_names_by_month,
    resolve_sample_name_column, validate_month_filter_args,
)
from sentinel_crop_seg.core.evaluation.metrics import (
    compute_iou_from_confusion_matrix,
    compute_mean_class_accuracy_from_confusion_matrix,
    compute_mean_iou_from_confusion_matrix,
    compute_per_class_accuracy_from_confusion_matrix,
    compute_pixel_accuracy_from_confusion_matrix,
    update_confusion_matrix_from_predictions,
)
from sentinel_crop_seg.core.evaluation.reporting import (
    DEFAULT_CLASS_NAMES, save_confusion_matrices, save_evaluated_samples_txt,
    save_evaluation_report, save_per_class_metrics_csv,
    save_summary_json, save_summary_txt,
)
