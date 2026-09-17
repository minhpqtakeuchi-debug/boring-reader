from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from typing import Union, Optional, List, Dict, Any

# ==========================
# Pydantic schema for output
# ==========================

BlowValue = Union[float, List[float], None]


class HeaderInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    investigation_name: Optional[str] = None
    project_name: Optional[str] = None

    # required-but-nullable -> Optional[...] without default None means
    # "key must exist but may be null" in a *full* log; for partial logs we
    # let them be optional at the BoreholeLog level (see below).
    borehole_no: Optional[str] = None
    sheet_no: Optional[float] = None
    location: Optional[str] = None
    client: Optional[str] = None
    contractor: Optional[str] = None
    chief_engineer: Optional[str] = None

    investigation_period_start: Optional[str] = None
    investigation_period_end: Optional[str] = None

    collar_elevation_el_m: Optional[float] = None
    total_depth_m: Optional[float] = None

    drilling_machine: Optional[str] = None
    engine: Optional[str] = None
    hammer_device: Optional[str] = None
    pump: Optional[str] = None


class SoilLayer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_depth_m: float
    end_depth_m: float
    layer_thickness_m: float
    soil_classification: Optional[str]  # required in schema but may be null
    color: Optional[str] = None
    description: Optional[str] = None


class SptTest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_depth_range_m: List[float]
    blow_counts_0_10cm: BlowValue = None
    blow_counts_10_20cm: BlowValue = None
    blow_counts_20_30cm: BlowValue = None
    penetration_cm: Optional[float] = None
    n_value: Optional[float] = None

    @field_validator("test_depth_range_m")
    @classmethod
    def _check_depth_range(cls, v: List[float]) -> List[float]:
        if not (1 <= len(v) <= 2):
            raise ValueError("test_depth_range_m must have length 1 or 2")
        return v

    @field_validator(
        "blow_counts_0_10cm",
        "blow_counts_10_20cm",
        "blow_counts_20_30cm",
    )
    @classmethod
    def _check_blow_counts(cls, v: BlowValue) -> BlowValue:
        if v is None:
            return v
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("blow_count array must have exactly 2 numbers")
            return [float(x) for x in v]
        raise TypeError("blow_counts must be number, 2-number list, or null")


class BoreholeLogPartial(BaseModel):
    """
    Partial borehole_log:

    - header_info may exist or not
    - soil_layers may exist or not
    - spt_tests may exist or not

    This covers:
      * header-only JSON
      * soil-only JSON
      * spt-only JSON
      * full-borelog JSON
    """
    model_config = ConfigDict(extra="forbid")

    header_info: Optional[HeaderInfo] = None
    soil_layers: Optional[List[SoilLayer]] = None
    spt_tests: Optional[List[SptTest]] = None


class BoreholeLogEnvelope(BaseModel):
    """
    Top-level object: { "borehole_log": { ...partial... } }
    Extra keys like "$schema", "title", etc. are ignored.
    """
    model_config = ConfigDict(extra="allow")

    borehole_log: BoreholeLogPartial
