"""Storage-specific API failures; --yes never bypasses them."""

from sparkrun.api._errors import SparkrunError


class InvalidModelSelection(SparkrunError, ValueError):
    pass


class ModelInUse(SparkrunError):
    pass


class StorageUnreachable(SparkrunError):
    pass


class StorageCapabilityMissing(SparkrunError):
    pass


class UnsafeStoragePath(SparkrunError):
    pass


class RemovalPlanChanged(SparkrunError):
    pass
