from abc import ABC, ABCMeta

from metrics.base import DataStructure


class MetricDataParserBase(ABC):
    @classmethod
    def parse_build_data(cls, data):
        """
        Args:
            data: dictionary which will be passed to InputBuildData
        """
        data = cls._validate_build_data(data)
        data = cls._process_build_data(data)
        return data

    @classmethod
    def parse_non_tensor_data(cls, data):
        """
        Args:
            data: dictionary which will be passed to InputDataStructure
        """
        input_data = cls._validate_non_tensor_data(data)
        output_data = cls._process_non_tensor_data(input_data)
        return output_data

    @classmethod
    def _validate_build_data(cls, data):
        """
        Specify assertions that tensor data should contains

        Args:
            data: dictionary
        Return:
            InputDataStructure
        """
        return cls.InputBuildData(data)

    @classmethod
    def _validate_non_tensor_data(cls, data):
        """
        Specify assertions that non-tensor data should contains

        Args:
            data: dictionary
        Return:
            InputDataStructure
        """
        return cls.InputNonTensorData(data)

    """
    Override these two functions if needed.
    """
    @classmethod
    def _process_build_data(cls, data):
        """
        Process data in order to following metrics can use it

        Args:
            data: InputBuildData

        Return:
            OutputBuildData
        """
        # default function is just passing data
        return cls.OutputBuildData(data.to_dict())

    @classmethod
    def _process_non_tensor_data(cls, data):
        """
        Process data in order to following metrics can use it

        Args:
            data: InputNonTensorData

        Return:
            OutputNonTensorData
        """
        # default function is just passing data
        return cls.OutputNonTensorData(data.to_dict())

    """
    Belows should be implemented when inherit.
    """
    class InputBuildData(DataStructure, metaclass=ABCMeta):
        pass

    class OutputBuildData(DataStructure, metaclass=ABCMeta):
        pass

    class InputNonTensorData(DataStructure, metaclass=ABCMeta):
        pass

    class OutputNonTensorData(DataStructure, metaclass=ABCMeta):
        pass


class AudioDataParser(MetricDataParserBase):
    class InputBuildData(DataStructure):
        _keys = [
            "dataset_split_name",
            "label_names",
            "losses",  # Dict | loss_key -> Tensor
            "learning_rate",
            "wavs",
        ]

    class OutputBuildData(DataStructure):
        _keys = [
            "dataset_split_name",
            "label_names",
            "losses",
            "learning_rate",
            "wavs",
        ]

    class InputNonTensorData(DataStructure):
        _keys = [
            "dataset_split_name",
            "label_names",
            "predictions_onehot",
            "labels_onehot",
        ]

    class OutputNonTensorData(DataStructure):
        _keys = [
            "dataset_split_name",
            "label_names",
            "predictions_onehot",
            "labels_onehot",
            "predictions",
            "labels",
        ]

    @classmethod
    def _process_non_tensor_data(cls, data):
        predictions = data.predictions_onehot.argmax(axis=-1)
        labels = data.labels_onehot.argmax(axis=-1)

        return cls.OutputNonTensorData({
            "dataset_split_name": data.dataset_split_name,
            "label_names": data.label_names,
            "predictions_onehot": data.predictions_onehot,
            "labels_onehot": data.labels_onehot,
            "predictions": predictions,
            "labels": labels,
        })
