from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix
)

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, digits=4)
    }
