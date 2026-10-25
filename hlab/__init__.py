"""HandwritingLab core package."""

import warnings

# scikit-learn's joblib/parallel layer emits a noisy UserWarning during some
# predict calls; it's harmless and would otherwise spam the console.
warnings.filterwarnings(
    "ignore", message=".*sklearn.utils.parallel.delayed.*", category=UserWarning)
