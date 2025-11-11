"""Custom exceptions for PartiTone."""


class PartiToneError(Exception):
    """Base exception for PartiTone errors."""
    pass


class AudioProcessingError(PartiToneError):
    """Raised when audio processing fails."""
    pass


class StemNotFoundError(PartiToneError):
    """Raised when a required stem file is not found."""
    pass


class InvalidAudioFormatError(PartiToneError):
    """Raised when an unsupported audio format is provided."""
    pass


class DemucsExecutionError(PartiToneError):
    """Raised when Demucs subprocess execution fails."""
    pass


class FileSizeLimitError(PartiToneError):
    """Raised when uploaded file exceeds size limit."""
    pass


class FileReadError(PartiToneError):
    """Raised when file reading fails."""
    pass


class FileWriteError(PartiToneError):
    """Raised when file writing fails."""
    pass


class ModelInitializationError(PartiToneError):
    """Raised when model initialization fails."""
    pass


class JobNotFoundError(PartiToneError):
    """Raised when a job is not found."""
    pass

