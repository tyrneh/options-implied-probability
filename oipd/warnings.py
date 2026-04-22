"""Public warning taxonomy for OIPD."""


class OIPDWarning(UserWarning):
    """Base class for warning categories emitted by OIPD."""


class DataQualityWarning(OIPDWarning):
    """Warning for issues in observed input data quality."""


class ModelRiskWarning(OIPDWarning):
    """Warning for economically questionable fitted model outputs."""


class NumericalWarning(OIPDWarning):
    """Warning for numerical fragility or instability."""


class WorkflowWarning(OIPDWarning):
    """Warning for partial-operation or continuation behavior."""


__all__ = [
    "OIPDWarning",
    "DataQualityWarning",
    "ModelRiskWarning",
    "NumericalWarning",
    "WorkflowWarning",
]
