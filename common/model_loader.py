from typing import List

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import gfile

from common.utils import format_text, get_logger


class Ckpt():
    def __init__(
        self,
        session: tf.Session,
        variables_to_restore=None,
        include_scopes: str="",
        exclude_scopes: str="",
        ignore_missing_vars: bool=False,
        use_ema: bool=False,
        ema_decay: float=None,
        logger=None,
    ):
        self.session = session
        self.variables_to_restore = self._get_variables_to_restore(
            variables_to_restore,
            include_scopes,
            exclude_scopes,
            use_ema,
            ema_decay,
        )
        self.ignore_missing_vars = ignore_missing_vars
        self.logger = logger
        if logger is None:
            self.logger = get_logger("Ckpt Loader")

        # variables to save reusable info from previous load
        self.has_previous_info = False
        self.grouped_vars = {}
        self.placeholders = {}
        self.assign_op = None

    def _get_variables_to_restore(
        self,
        variables_to_restore=None,
        include_scopes: str="",
        exclude_scopes: str="",
        use_ema: bool=False,
        ema_decay: float=None,
    ):
        # variables_to_restore might be List or Dictionary.

        def split_strip(scopes: str):
            return list(filter(lambda x: len(x) > 0, [s.strip() for s in scopes.split(",")]))

        def starts_with(var, scopes: List) -> bool:
            return any([var.op.name.startswith(prefix) for prefix in scopes])

        exclusions = split_strip(exclude_scopes)
        inclusions = split_strip(include_scopes)

        if variables_to_restore is None:
            if use_ema:
                if ema_decay is None:
                    raise ValueError("ema_decay undefined")
                else:
                    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
                    variables_to_restore = ema.variables_to_restore()  # dictionary
            else:
                variables_to_restore = tf.contrib.framework.get_variables_to_restore()

        filtered_variables_key = variables_to_restore
        if len(inclusions) > 0:
            filtered_variables_key = filter(lambda var: starts_with(var, inclusions), filtered_variables_key)
        filtered_variables_key = filter(lambda var: not starts_with(var, exclusions), filtered_variables_key)

        if isinstance(variables_to_restore, dict):
            variables_to_restore = {
                key: variables_to_restore[key] for key in filtered_variables_key
            }
        elif isinstance(variables_to_restore, list):
            variables_to_restore = list(filtered_variables_key)

        return variables_to_restore

    # Copied and revised code not to create duplicated 'assign' operations everytime it gets called.
    # https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/contrib/framework/python/ops/variables.py#L558
    def load(self, checkpoint_stempath):
        def get_variable_full_name(var):
            if var._save_slice_info:
                return var._save_slice_info.full_name
            else:
                return var.op.name

        if not self.has_previous_info:
            if isinstance(self.variables_to_restore, (tuple, list)):
                for var in self.variables_to_restore:
                    ckpt_name = get_variable_full_name(var)
                    if ckpt_name not in self.grouped_vars:
                        self.grouped_vars[ckpt_name] = []
                    self.grouped_vars[ckpt_name].append(var)

            else:
                for ckpt_name, value in self.variables_to_restore.items():
                    if isinstance(value, (tuple, list)):
                        self.grouped_vars[ckpt_name] = value
                    else:
                        self.grouped_vars[ckpt_name] = [value]

        # Read each checkpoint entry. Create a placeholder variable and
        # add the (possibly sliced) data from the checkpoint to the feed_dict.
        reader = pywrap_tensorflow.NewCheckpointReader(str(checkpoint_stempath))
        feed_dict = {}
        assign_ops = []
        for ckpt_name in self.grouped_vars:
            if not reader.has_tensor(ckpt_name):
                log_str = f"Checkpoint is missing variable [{ckpt_name}]"
                if self.ignore_missing_vars:
                    self.logger.warning(log_str)
                    continue
                else:
                    raise ValueError(log_str)
            ckpt_value = reader.get_tensor(ckpt_name)

            for var in self.grouped_vars[ckpt_name]:
                placeholder_name = f"placeholder/{var.op.name}"
                if self.has_previous_info:
                    placeholder_tensor = self.placeholders[placeholder_name]
                else:
                    placeholder_tensor = tf.placeholder(
                        dtype=var.dtype.base_dtype,
                        shape=var.get_shape(),
                        name=placeholder_name)
                    assign_ops.append(var.assign(placeholder_tensor))
                    self.placeholders[placeholder_name] = placeholder_tensor

                if not var._save_slice_info:
                    if var.get_shape() != ckpt_value.shape:
                        raise ValueError(
                            f"Total size of new array must be unchanged for {ckpt_name} "
                            f"lh_shape: [{str(ckpt_value.shape)}], rh_shape: [{str(var.get_shape())}]")

                    feed_dict[placeholder_tensor] = ckpt_value.reshape(ckpt_value.shape)
                else:
                    slice_dims = zip(var._save_slice_info.var_offset,
                                     var._save_slice_info.var_shape)
                    slice_dims = [(start, start + size) for (start, size) in slice_dims]
                    slice_dims = [slice(*x) for x in slice_dims]
                    slice_value = ckpt_value[slice_dims]
                    slice_value = slice_value.reshape(var._save_slice_info.var_shape)
                    feed_dict[placeholder_tensor] = slice_value

        if not self.has_previous_info:
            self.assign_op = control_flow_ops.group(*assign_ops)

        self.session.run(self.assign_op, feed_dict)

        if len(feed_dict) > 0:
            for key in feed_dict.keys():
                self.logger.info(f"init from checkpoint > {key}")
        else:
            self.logger.info(f"No init from checkpoint")

        with format_text("cyan", attrs=["bold", "underline"]) as fmt:
            self.logger.info(fmt(f"Restore from {checkpoint_stempath}"))
        self.has_previous_info = True
