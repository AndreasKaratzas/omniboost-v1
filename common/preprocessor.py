
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from common.io import save_pickle


# NOTE: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
def benchmarking_scaler(
    benchmarkings: np.ndarray,
    return_mix: bool = True
):
    """Scale the benchmarkings.

    Parameters
    ----------
    benchmarkings : np.ndarray
        The benchmarkings.
    return_mix : bool, optional
        Whether to return the scaled benchmarkings
    
    Returns
    -------
    np.ndarray
        The scaled benchmarkings.
    """
    sc = StandardScaler()
    norm = MinMaxScaler()
    X_train_std = sc.fit_transform(benchmarkings)
    X_train_std_norm = norm.fit_transform(X_train_std)
    save_pickle('../data/demo/StandardScaler.pkl', sc)
    save_pickle('../data/demo/MinMaxScaler.pkl', norm)
    if return_mix:
        return X_train_std_norm
    else:
        return X_train_std
