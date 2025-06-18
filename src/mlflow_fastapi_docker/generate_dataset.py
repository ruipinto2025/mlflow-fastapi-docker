from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class UnsupportedNoiseTypeError(ValueError):
    def __init__(self, noise_type: str):
        super().__init__(f"Noise type '{noise_type}' is not supported.")


def load_and_preprocess_data(filenames: list[str]) -> tuple[pd.DataFrame, StandardScaler, list[str]]:
    """
    Loads Cubestocker normal data, removes unnecessary features, initializes a new column 'Anomaly_Label' as 0 (normal data) and scales the data.

    Args:
        filenames: Names of the Cubestocker files to be loaded

    Returns:
        Preprocessed Dataframe, the scaler, and a list of features used for training.
    """
    dfs = []
    for fname in filenames:
        dfi = pd.read_parquet(f"data/{fname}.parquet", engine="fastparquet")
        dfi = dfi.drop(columns=["new_stamp", "device_id", "date"], errors="ignore")
        dfi["Anomaly_Label"] = 0
        dfs.append(dfi)

    df = pd.concat(dfs, ignore_index=True)

    feature_cols = [c for c in df.columns if c != "Anomaly_Label"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler, feature_cols


def _get_noise_columns(feature_cols: list[str], exclude_cols: list[str]) -> list[str]:
    """Filter out excluded columns from feature columns."""
    return [col for col in feature_cols if col not in exclude_cols]


def add_noise(
    df: pd.DataFrame, noise_pct: float, noise_std: float, noise_type: str, feature_cols: list[str]
) -> pd.DataFrame:
    """
    Add synthetic noise to data to create anomalies.

    Args:
        df: Input DataFrame
        noise_pct: Percentage of rows to add noise to
        noise_std: Standard deviation of the noise distribution
        noise_type: Type of noise to add (gaussian, laplace, poisson, etc.)
        feature_cols: Feature columns to be added noise

    Returns:
        A new DataFrame where `noise_pct * len(df)` randomly chosen rows have had their feature values perturbed according to `noise_type` & `noise_std`. For those rows 'Anomaly_Label' is set to 1.
    """
    # add seed for reproducibility
    rng = np.random.RandomState(0)
    num_rows = len(df)
    num_noise = int(num_rows * noise_pct)
    noise_idx = rng.choice(df.index, size=num_noise, replace=False)

    # -- classical noise types --
    # gaussian, laplace, poisson, gamma, bernoulli
    noise_mean = 0

    exclude_cols = [
        "cr1_mode",
        "cr2_mode",
        "cr1_has_request",
        "cr2_has_request",
        "cr1_has_carrier",
        "cr2_has_carrier",
        "cr1_to_exit_full",
        "cr2_to_exit_full",
    ]

    noise_columns = _get_noise_columns(feature_cols, exclude_cols)

    if noise_type == "gaussian":
        noise = rng.normal(noise_mean, noise_std, size=(num_noise, len(noise_columns)))
    elif noise_type == "laplace":
        noise = rng.laplace(noise_mean, noise_std, size=(num_noise, len(noise_columns)))
    elif noise_type == "poisson":
        noise = rng.poisson(noise_std, size=(num_noise, len(noise_columns))) - noise_std
    elif noise_type == "gamma":
        shape = 1
        scale = noise_std / np.sqrt(shape)
        noise = rng.gamma(shape, scale, size=(num_noise, len(noise_columns)))
        # Center the gamma noise
        noise = noise - noise.mean()
        if noise.std() != 0:
            noise *= noise_std / noise.std()
    elif noise_type == "bernoulli":
        # Flip sign of ~50% of the features in those rows
        mask = rng.binomial(1, 0.5, size=(num_noise, len(noise_columns))).astype(bool)
        for idx, row_mask in zip(noise_idx, mask):
            for j, col in enumerate(noise_columns):
                if row_mask[j]:
                    df.loc[idx, col] = -df.loc[idx, col]
        df.loc[noise_idx, "Anomaly_Label"] = 1
        return df
    else:
        raise UnsupportedNoiseTypeError(noise_type)

    # Add the continuous noise for the non-bernoulli types:
    df.loc[noise_idx, noise_columns] += noise
    df.loc[noise_idx, "Anomaly_Label"] = 1
    return df


# Temperature limits (Float, ºC)
TEMP_LIMITS: dict[str, tuple[float, float]] = {"x": (20.00, 25.00), "y": (20.00, 70.00), "z": (20.00, 22.00)}

# Current limits (Float, A)
CURRENT_LIMITS: dict[str, tuple[float, float]] = {"x": (0.00, 20.00), "y": (0.00, 10.00), "z": (0.00, 5.00)}

# Speed limits (Float, RPM)
SPEED_LIMITS: dict[str, tuple[float, float]] = {"x": (-2000, 2000.00), "y": (-3000, 3000.00), "z": (-1500, 1500.00)}

# Position limits (Integer, mm)
POSITION_LIMITS: dict[str, tuple[int, int]] = {"x": (2148, 41755), "y": (-902, 432400), "z": (-70901, 70901)}

# Torque limits (Float, Nm)
TORQUE_LIMITS: dict[str, tuple[float, float]] = {"x": (-60.00, 55.00), "y": (-15.00, 15.00), "z": (-5.00, 5.00)}

# Power limits (Float, kWh)
POWER_LIMITS: dict[str, tuple[float, float]] = {"x": (-5.5, 5.5), "y": (-2.85, 2.85), "z": (-0.4, 0.4)}

# Belt vibration angle limits (Float, °)
ANGLE_X_LIMITS: tuple[float, float] = (174.00, 186.00)
ANGLE_Y_LIMITS: tuple[float, float] = (74.00, 106.00)


def check_motor_overheating(row: dict[str, Any], prefix: str = "cr2_") -> str | None:
    """
    Motor Overheating:
    If the temperature exceeds its maximum in any axis, flag a Motor Overheating anomaly.
    """
    for axis in ["x", "y", "z"]:
        temp = row[f"{prefix}mt{axis}_temperature"]
        _, temp_max = TEMP_LIMITS[axis]
        if temp > temp_max:
            return f"Motor Overheating ({prefix}mtx/mty/mtz)"
    return None


def check_motor_overcurrent(row: dict[str, Any], prefix: str = "cr2_") -> str | None:
    """
    Motor Overcurrent
    If the current exceeds its maximum in any axis, flag a Motor Overcurrent anomaly.
    """
    for axis in ["x", "y", "z"]:
        current = row[f"{prefix}mt{axis}_current"]
        _, curr_max = CURRENT_LIMITS[axis]
        if current > curr_max:
            return f"Motor Overcurrent ({prefix}mtx/mty/mtz)"
    return None


def check_anomalous_motor_position(row: dict[str, Any], prefix: str = "cr2_") -> str | None:
    """
    Anomalous Motor Position:
    If the motor position is out of its allowed range, flag the anomaly "anomalous motor position".
    """
    for axis in ["x", "y", "z"]:
        pos = row[f"{prefix}mt{axis}_position"]
        pos_min, pos_max = POSITION_LIMITS[axis]

        if pos < pos_min or pos > pos_max:
            return f"Anomalous Motor Position ({prefix}mtx/mty/mtz)"
    return None


def check_high_speed_operation(row: dict[str, Any], prefix: str = "cr2_") -> str | None:
    """
    High Speed Operation:
    If the speed in any axis is outside the allowed range while the current remains within its normal range,
    flag an High Speed Operation anomaly.
    """
    for axis in ["x", "y", "z"]:
        speed = row[f"{prefix}mt{axis}_speed"]
        s_min, s_max = SPEED_LIMITS[axis]
        if speed < s_min or speed > s_max:
            return f"High Speed Operation ({prefix}mtx/mty/mtz)"
    return None


def check_abnormal_motor_torque(row: dict[str, Any], prefix: str = "cr2_") -> str | None:
    """
    Abnormal Motor Torque:
    If the torque in any axis deviates beyond its allowed range flag an Abnormal Motor Torque anomaly.
    """
    for axis in ["x", "y", "z"]:
        torque = row[f"{prefix}mt{axis}_torque"]
        t_min, t_max = TORQUE_LIMITS[axis]
        if torque < t_min or torque > t_max:
            return f"Abnormal Motor Torque ({prefix}mtx/mty/mtz)"
    return None


def check_abnormal_power_variation(row: dict[str, Any], prefix: str = "cr2_") -> str | None:
    """
    Abnormal Power Variation:
    If the power reading in any axis is outside its normal range flag a Abnormal Power Variation anomaly.
    """
    for axis in ["x", "y", "z"]:
        power = row[f"{prefix}mt{axis}_power"]
        p_min, p_max = POWER_LIMITS[axis]
        if power < p_min or power > p_max:
            return f"Abnormal Power Variation ({prefix}mtx/mty/mtz)"
    return None


def check_high_vibration(row: dict[str, Any], prefix: str = "cr2_") -> str | None:
    """
    High Vibration:
    If the vibration angles (from belt sensors) are outside their permitted ranges flag a High Vibration anomaly.
    """

    xangle = row[f"{prefix}y_beltvibration_xangle"]
    yangle = row[f"{prefix}y_beltvibration_yangle"]
    excessive_vibration = (
        xangle < ANGLE_X_LIMITS[0]
        or xangle > ANGLE_X_LIMITS[1]
        or yangle < ANGLE_Y_LIMITS[0]
        or yangle > ANGLE_Y_LIMITS[1]
    )

    if excessive_vibration:
        return f"High Vibration ({prefix}y_beltvibration)"
    return None


ANOMALY_DETAILS_MAP: dict[str, dict[str, Any]] = {
    "Motor Overcurrent": {
        "severity_label": "Critical",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "Current spikes outside the normal range characterize a critical overload.",
    },
    "Motor Overheating": {
        "severity_label": "Critical",
        "action": "Activate the cooling protocol",
        "explanation": "Temperature has exceeded the normal value, indicating overheating.",
    },
    "Anomalous Motor Position": {
        "severity_label": "Critical",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "The motor position is outside the normal range, possibly due to excessive speed.",
    },
    "High Speed Operation": {
        "severity_label": "Critical",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "Speed exceeds the normal limit, suggesting a possible malfunction or control failure.",
    },
    "Abnormal Motor Torque": {
        "severity_label": "Critical",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "The force applied to the motor is outside the normal range.",
    },
    "High Vibration": {
        "severity_label": "High",
        "action": "Perform a mechanical inspection of the crane; check for wear and angular alignment",
        "explanation": "Vibration angles deviate significantly from normal values, indicating high vibration.",
    },
    "Abnormal Power Variation": {
        "severity_label": "Medium",
        "action": "Stop immediately to prevent damage and collisions. Reduce operation intensity and analyze whether system configuration adjustments are needed",
        "explanation": "Motor power is outside its normal values.",
    },
}


def decision_layer(anomaly_labels: list[str]) -> list[dict[str, Any]]:
    """
    Prioritize anomalies based on the following order:
      1. Motor Overcurrent (Critical)
      2. Motor Overheating (Critical)
      3. Anomalous Motor Position (Critical)
      4. High Speed Operation (Critical)
      5. Abnormal Motor Torque (Critical)
      6. High Vibration (High)
      7. Abnormal Power Variation (Medium)

    Returns a list with a single dictionary containing details for the highest priority anomaly.
    """

    priority_order = [
        "Motor Overcurrent",
        "Motor Overheating",
        "Anomalous Motor Position",
        "High Speed Operation",
        "Abnormal Motor Torque",
        "High Vibration",
        "Abnormal Power Variation",
    ]

    for anomaly_type in priority_order:
        matching = [label for label in anomaly_labels if anomaly_type in label]
        if matching:
            info = ANOMALY_DETAILS_MAP[anomaly_type]
            return [
                {
                    "anomaly": matching[0],
                    "severity": info["severity_label"],
                    "action": info["action"],
                    "explanation": info["explanation"],
                }
            ]
    return []


def get_anomaly_labels(row: pd.Series) -> list[str]:
    """
    Apply domain-specific checks for both Crane 1 (prefix 'cr1_') and Crane 2 (prefix 'cr2_'),
    and return a list of anomaly labels detected.
    """
    labels = []
    data = row.to_dict()
    for prefix in ["cr1_", "cr2_"]:
        for check in (
            check_motor_overcurrent,
            check_motor_overheating,
            check_anomalous_motor_position,
            check_high_speed_operation,
            check_abnormal_power_variation,
            check_abnormal_motor_torque,
            check_high_vibration,
        ):
            msg = check(data, prefix=prefix)
            if msg is not None:
                labels.append(msg)
    return labels


def get_numerical_anomaly_label(row: pd.Series, mapping: dict[str, int]) -> int:
    symbolic_labels = row["symbolic_labels"]

    if not symbolic_labels:
        return mapping["Normal"]
    else:
        # If symbolic labels are found, apply the priority logic
        priority_order = [
            "Motor Overcurrent",
            "Motor Overheating",
            "Anomalous Motor Position",
            "High Speed Operation",
            "Abnormal Motor Torque",
            "High Vibration",
            "Abnormal Power Variation",
        ]
        for anomaly_type in priority_order:
            matching_labels = [label for label in symbolic_labels if anomaly_type in label]
            if matching_labels:
                return mapping[anomaly_type]
        # Fallback in case a symbolic label was generated but didn't match priority_order (shouldn't happen)
        return mapping["Normal"]


def generate_dataset_with_anomalies(
    noise_pct: float, noise_std: float, noise_type: str
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """
    Generate a dataset with synthetic anomalies and proper anomaly type labels.

    Args:
        noise_pct: Percentage of rows to add noise to
        noise_std: Standard deviation of the noise distribution
        noise_type: Type of noise to add (gaussian, laplace, poisson, etc.)

    Returns:
        - Dataset with anomalies
        - DataFrame with features and Anomaly_Type_Label
        - Mapping from anomaly type names to integers
    """
    # 1) Load & add noise
    filenames = [
        "TRACKING_20250502_a_20250508",
        "TRACKING_20250509_a_20250515",
        "TRACKING_20250516_a_20250522",
        "TRACKING_20250523_a_20250529",
    ]
    df, scaler, feature_cols = load_and_preprocess_data(filenames)
    df = add_noise(df, noise_pct, noise_std, noise_type, feature_cols)

    # 2) Reverse-scale & symbolic labeling
    reversed_data = scaler.inverse_transform(df[feature_cols])
    reversed_df = pd.DataFrame(reversed_data, columns=feature_cols)
    reversed_df["Anomaly_Label"] = df["Anomaly_Label"]
    reversed_df["symbolic_labels"] = reversed_df.apply(get_anomaly_labels, axis=1)

    # 3) Build anomaly-type → int mapping
    all_anomaly_types = set()
    for labels_list in reversed_df["symbolic_labels"]:
        for label in labels_list:
            # Extract only the anomaly type string, excluding the parenthesized part
            anomaly_type = label.split(" (")[0]
            all_anomaly_types.add(anomaly_type)

    sorted_anomaly_types = sorted(all_anomaly_types)
    anomaly_type_mapping = {anom_type: i + 1 for i, anom_type in enumerate(sorted_anomaly_types)}
    anomaly_type_mapping["Normal"] = 0  # Add label for normal data

    # 4) Create numeric labels
    reversed_df["Anomaly_Type_Label"] = 0  # Initialize with 0 (Normal)
    reversed_df["Anomaly_Type_Label"] = reversed_df.apply(
        lambda row: get_numerical_anomaly_label(row, anomaly_type_mapping), axis=1
    )
    df = reversed_df.drop(columns=["symbolic_labels", "Anomaly_Label"])

    feature_cols = [c for c in df.columns if c != "Anomaly_Type_Label"]

    return df, feature_cols, anomaly_type_mapping
