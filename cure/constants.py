"""Shared degradation vocabularies.

The order is part of the legacy checkpoint and HDF5 formats. Do not reorder it.
"""

EMBEDDER_TYPES = (
    "clear",
    "low",
    "haze",
    "rain",
    "snow",
    "low_haze",
    "low_rain",
    "low_snow",
    "haze_rain",
    "haze_snow",
    "low_haze_rain",
    "low_haze_snow",
)

# The CURE fine-tuning stage excludes triple degradations (paper, Sec. 3.4).
RESTORATION_TYPES = (
    "low",
    "haze",
    "rain",
    "snow",
    "low_haze",
    "low_rain",
    "low_snow",
    "haze_rain",
    "haze_snow",
)

# The OneRestore baseline sees every non-clean CCDD-11 class.
BASELINE_TYPES = EMBEDDER_TYPES[1:]

IDENTITY_NAME = "identity"
EMBEDDING_DIM = 324
