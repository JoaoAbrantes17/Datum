class DatumError(Exception):
    """Base for all Datum errors."""


class DatumFetchError(DatumError):
    """Provider network or API failure."""


class DatumSchemaError(DatumError):
    """Unexpected column shape or dtype from provider."""


class DatumMissingDataError(DatumError):
    """Missing data detected under strict dropna_policy."""
