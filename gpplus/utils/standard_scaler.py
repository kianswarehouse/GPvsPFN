import torch


class StandardScaler:
    """A utility class for standardizing data by subtracting the mean and dividing by the standard deviation.

    This scaler follows a similar API to scikit-learn's StandardScaler but is implemented
    for PyTorch tensors. It computes the mean and standard deviation of the features
    along the first dimension (assuming data shape is [N, features]).

    Attributes:
        mean (torch.Tensor or None): The per-feature mean, computed in the `fit` method.
        std (torch.Tensor or None): The per-feature standard deviation, computed in the `fit` method.
    """

    def __init__(self):
        """Initializes a new instance of the StandardScaler with empty mean and std attributes."""
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor) -> None:
        """Computes the mean and standard deviation for each feature in the given data.

        Args:
            data (torch.Tensor): The input data of shape [N, features].

        Notes:
            - If `data` has NaN or Inf values, the computed statistics may be invalid.
            - Uses `std(dim=0, correction=0)` which mimics `unbiased=False` behavior (like
              dividing by N instead of N-1 in NumPy).
        """
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, correction=0, keepdim=True)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Applies standardization to the input data using the stored mean and std.

        Args:
            data (torch.Tensor): The input data of shape [N, features] to be standardized.

        Returns:
            torch.Tensor: The standardized data, where each feature has zero mean and unit variance.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted. Call `fit` first.")

        # (Optional) You may want to handle the case where self.std == 0, which can lead to NaNs.
        # For instance:
        # safe_std = torch.where(self.std == 0, torch.ones_like(self.std), self.std)
        # return (data - self.mean) / safe_std

        return (data - self.mean) / self.std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverts the standardization of the input data using the stored mean and std.

        Args:
            data (torch.Tensor): The standardized data of shape [N, features].

        Returns:
            torch.Tensor: The data in its original scale.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted. Call `fit` first.")
        return data * self.std + self.mean


class LogScaler:
    """A utility class for standardizing data in log space.

    This scaler first applies a log transformation to the data, then standardizes
    the log-transformed values. This is useful for heavily right-skewed distributions
    or when the coefficient of variation (std/mean) is high.

    The transformation pipeline is:
    1. Log transform: log(data + epsilon) where epsilon prevents log(0)
    2. Standardize: (log_data - mean) / std

    The inverse transformation pipeline is:
    1. Unstandardize: log_data = standardized * std + mean
    2. Exp transform: exp(log_data) - epsilon

    Attributes:
        mean (torch.Tensor or None): The per-feature mean of log-transformed data, computed in `fit`.
        std (torch.Tensor or None): The per-feature standard deviation of log-transformed data, computed in `fit`.
        min (torch.Tensor or None): The per-feature minimum value of original data, computed in `fit`.
        epsilon (float): Small value added before log transform to avoid log(0). Default is 1e-8.
    """

    def __init__(self, epsilon: float = 1e-8):
        """Initializes a new instance of the LogScaler.

        Args:
            epsilon (float): Small value added to data before log transform to avoid log(0).
                Default is 1e-8.
        """
        self.mean = None
        self.std = None
        self.min = None
        self.epsilon = epsilon

    def fit(self, data: torch.Tensor) -> None:
        """Computes the mean and standard deviation for log-transformed data.

        Args:
            data (torch.Tensor): The input data of shape [N, features]. Should contain
                positive values (or values that become positive after adding epsilon).

        Notes:
            - Applies log(data + epsilon) before computing statistics.
            - If `data` has NaN or Inf values, the computed statistics may be invalid.
            - Uses `std(dim=0, correction=0)` which mimics `unbiased=False` behavior.
            - Warns if data contains negative values that would become problematic.
        """
        # Check if we have negative values (due to noise)
        data_min = data.min(dim=0, keepdim=True)[0]
        has_negatives = torch.any(data_min < 0)
        
        if has_negatives:
            # Store minimum value and shift data to be non-negative
            self.min = data_min
            data_shifted = data - self.min + self.epsilon
        else:
            # No negatives, no need to shift
            self.min = None
            data_shifted = data + self.epsilon

        # Apply log transform
        log_data = torch.log(data_shifted)

        # Compute statistics on log-transformed data
        self.mean = log_data.mean(dim=0, keepdim=True)
        self.std = log_data.std(dim=0, correction=0, keepdim=True)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Applies log transformation and standardization to the input data.

        Args:
            data (torch.Tensor): The input data of shape [N, features] to be transformed.

        Returns:
            torch.Tensor: The log-transformed and standardized data, where each feature
                has zero mean and unit variance in log space.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("LogScaler has not been fitted. Call `fit` first.")

        # Shift data to be non-negative (only if we had negatives during fit)
        if self.min is not None:
            data_shifted = data - self.min + self.epsilon
        else:
            data_shifted = data + self.epsilon

        # Apply log transform
        log_data = torch.log(data_shifted)

        # Handle the case where self.std == 0, which can lead to NaNs
        zero_std_mask = self.std == 0
        if torch.any(zero_std_mask):
            import warnings
            # Handle both single-feature and multi-feature cases
            zero_std_squeezed = zero_std_mask.squeeze()
            if zero_std_squeezed.dim() == 0:
                # Single feature case
                zero_std_indices = [0] if zero_std_squeezed.item() else []
            else:
                # Multi-feature case
                zero_std_indices = torch.where(zero_std_squeezed)[0].tolist()
            warnings.warn(
                f"LogScaler: Standard deviation is zero for feature(s) {zero_std_indices}. "
                "This indicates constant values in log space. Using std=1.0 for these features to avoid division by zero."
            )
        safe_std = torch.where(zero_std_mask, torch.ones_like(self.std), self.std)

        # Standardize
        return (log_data - self.mean) / safe_std

    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Reverts the log transformation and standardization.

        Args:
            data (torch.Tensor): The standardized log-transformed data of shape [N, features].

        Returns:
            torch.Tensor: The data in its original scale.

        Raises:
            ValueError: If `mean` or `std` have not been set (i.e., if `fit` has not been called).
        """
        if self.mean is None or self.std is None:
            raise ValueError("LogScaler has not been fitted. Call `fit` first.")

        # Handle the case where self.std == 0
        zero_std_mask = self.std == 0
        if torch.any(zero_std_mask):
            import warnings
            # Handle both single-feature and multi-feature cases
            zero_std_squeezed = zero_std_mask.squeeze()
            if zero_std_squeezed.dim() == 0:
                # Single feature case
                zero_std_indices = [0] if zero_std_squeezed.item() else []
            else:
                # Multi-feature case
                zero_std_indices = torch.where(zero_std_squeezed)[0].tolist()
            warnings.warn(
                f"LogScaler: Standard deviation is zero for feature(s) {zero_std_indices} during inverse transform. "
                "This indicates constant values in log space. Using std=1.0 for these features to avoid division by zero."
            )
        safe_std = torch.where(zero_std_mask, torch.ones_like(self.std), self.std)

        # Unstandardize (reverse standardization)
        log_data = data * safe_std + self.mean

        # Apply inverse log transform (exp)
        exp_data = torch.exp(log_data)
        
        # Add back the minimum value if we shifted during fit
        if self.min is not None:
            return exp_data + self.min - self.epsilon
        else:
            return exp_data - self.epsilon


if __name__ == "__main__":
    # Example usage
    # -------------
    # Assume X_train is your training data as a torch.Tensor of shape [N, features].
    X_train = torch.randn(100, 10)  # example data

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_std = scaler.transform(X_train)

    # Later, for test data or predictions:
    X_test = torch.randn(20, 10)  # example test data
    X_test_std = scaler.transform(X_test)

    # Suppose predictions are in standardized space:
    predictions_std = torch.randn(20, 10)  # dummy model predictions
    predictions_original = scaler.inverse_transform(predictions_std)

    print("Standardized predictions:", predictions_std)
    print("Original-scale predictions:", predictions_original)

    # Example usage of LogScaler
    # --------------------------
    # LogScaler is useful for heavily right-skewed distributions
    # (e.g., buckling load, borehole flow rate)
    y_train = torch.abs(torch.randn(100, 1)) * 1000 + 100  # positive, right-skewed data

    log_scaler = LogScaler(epsilon=1e-8)
    log_scaler.fit(y_train)
    y_train_log_std = log_scaler.transform(y_train)

    # For test data or predictions:
    y_test = torch.abs(torch.randn(20, 1)) * 1000 + 100
    y_test_log_std = log_scaler.transform(y_test)

    # Suppose predictions are in standardized log space:
    predictions_log_std = torch.randn(20, 1)  # dummy model predictions
    predictions_original = log_scaler.inverse_transform(predictions_log_std)

    print("\nLogScaler example:")
    print("Original y_train range:", y_train.min().item(), "to", y_train.max().item())
    print("Log-standardized y_train range:", y_train_log_std.min().item(), "to", y_train_log_std.max().item())
    print("Reconstructed predictions range:", predictions_original.min().item(), "to", predictions_original.max().item())
