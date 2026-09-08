from sklearn.datasets import load_breast_cancer


def test_breast_cancer_shape():
    data = load_breast_cancer(as_frame=True)

    assert data.frame.shape == (569, 31)


def test_breast_cancer_classes():
    data = load_breast_cancer(as_frame=True)

    assert data.frame["target"].nunique() == 2


def test_breast_cancer_class_balance():
    data = load_breast_cancer(as_frame=True)

    class_balance = data.frame["target"].value_counts(normalize=True).min()

    assert class_balance >= 0.20
