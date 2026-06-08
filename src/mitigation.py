from sklearn.linear_model import LogisticRegression

from fairlearn.reductions import (
    ExponentiatedGradient,
    DemographicParity
)


def mitigate_bias(X_train, y_train, sensitive_feature):

    mitigator = ExponentiatedGradient(
        LogisticRegression(max_iter=5000),
        constraints=DemographicParity()
    )

    mitigator.fit(
        X_train,
        y_train,
        sensitive_features=sensitive_feature
    )

    return mitigator

    
