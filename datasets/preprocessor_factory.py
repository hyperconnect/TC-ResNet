from datasets.preprocessors import NoOpPreprocessor
from datasets.preprocessors import LogMelSpectrogramPreprocessor
from datasets.preprocessors import MFCCPreprocessor


_available_preprocessors = {
    # Audio
    "log_mel_spectrogram": LogMelSpectrogramPreprocessor,
    "mfcc": MFCCPreprocessor,
    # Noop
    "no_preprocessing": NoOpPreprocessor,
}


def factory(preprocess_method, scope, preprocessed_node_name):
    if preprocess_method in _available_preprocessors.keys():
        return _available_preprocessors[preprocess_method](scope, preprocessed_node_name)
    else:
        raise NotImplementedError(f"{preprocess_method}")
