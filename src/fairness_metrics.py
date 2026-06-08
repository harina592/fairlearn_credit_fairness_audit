from fairlearn.metrics import demographic_parity_difference


def calculate_fairness(y_true, y_pred, sensitive_feature):

    dp_difference = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive_feature
    )

    return dp_difference
