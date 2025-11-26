# logconf.py
def compute_metrics(conf_mat):
    """
    conf_mat[0,0] : TN
    conf_mat[0,1] : FP
    conf_mat[1,0] : FN
    conf_mat[1,1] : TP
    """
    tn, fp, fn, tp = conf_mat.ravel()

    accuracy = (tp + tn) / conf_mat.sum()

    if tp + fp == 0.0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0.0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1
