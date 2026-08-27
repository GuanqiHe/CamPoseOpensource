"""Kinematically equivalent Panda coordinate configurations.

Each configuration changes the sign convention of one joint coordinate in the
MJCF model.  Joint axis, range, initial position, state, and action must all be
transformed by the same sign.  The physical robot geometry is therefore
unchanged even though the robot model and action labels differ.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from robosuite.models.robots.manipulators.panda_robot import Panda
from robosuite.robots import register_robot_class
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class _EquivalentPandaBase(Panda):
    JOINT_SIGNS = np.ones(7, dtype=np.float64)
    CONFIG_ID = "cfg0"

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        signs = np.asarray(self.JOINT_SIGNS, dtype=np.float64)
        if signs.shape != (7,) or not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("JOINT_SIGNS must contain seven values in {-1, +1}")

        for joint, sign in zip(self._elements["joints"], signs):
            if sign > 0:
                continue
            axis = string_to_array(joint.get("axis"))
            joint.set("axis", array_to_string(-axis))
            if joint.get("range") is not None:
                lower, upper = string_to_array(joint.get("range"))
                joint.set("range", array_to_string(np.array([-upper, -lower])))

    @property
    def init_qpos(self):
        return np.asarray(self.JOINT_SIGNS) * super().init_qpos


@register_robot_class("FixedBaseRobot")
class PandaCfg0(_EquivalentPandaBase):
    JOINT_SIGNS = np.array([1, 1, 1, 1, 1, 1, 1])
    CONFIG_ID = "cfg0"


@register_robot_class("FixedBaseRobot")
class PandaCfg1(_EquivalentPandaBase):
    JOINT_SIGNS = np.array([-1, 1, 1, 1, 1, 1, 1])
    CONFIG_ID = "cfg1"


@register_robot_class("FixedBaseRobot")
class PandaCfg2(_EquivalentPandaBase):
    JOINT_SIGNS = np.array([1, -1, 1, 1, 1, 1, 1])
    CONFIG_ID = "cfg2"


@register_robot_class("FixedBaseRobot")
class PandaCfg3(_EquivalentPandaBase):
    JOINT_SIGNS = np.array([1, 1, -1, 1, 1, 1, 1])
    CONFIG_ID = "cfg3"


@register_robot_class("FixedBaseRobot")
class PandaCfg4(_EquivalentPandaBase):
    JOINT_SIGNS = np.array([1, 1, 1, -1, 1, 1, 1])
    CONFIG_ID = "cfg4"


CONFIG_SPECS = {
    cls.CONFIG_ID: {
        "robot": cls.__name__,
        "joint_signs": cls.JOINT_SIGNS.astype(int).tolist(),
    }
    for cls in (PandaCfg0, PandaCfg1, PandaCfg2, PandaCfg3, PandaCfg4)
}


def _register_sign_config(config_id: str, signs: list[int]):
    class_name = "Panda" + "".join(part.title() for part in config_id.split("_"))
    cls = type(
        class_name,
        (_EquivalentPandaBase,),
        {
            "__module__": __name__,
            "JOINT_SIGNS": np.asarray(signs, dtype=np.float64),
            "CONFIG_ID": config_id,
        },
    )
    cls = register_robot_class("FixedBaseRobot")(cls)
    globals()[class_name] = cls
    return cls


_SIGN_DR_DESIGN_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "joint_sign_dr_v1.json"
)
_SIGN_DR_DESIGN = json.loads(_SIGN_DR_DESIGN_PATH.read_text())


def _build_sign_dr_specs(split: str) -> dict[str, dict]:
    specs = {}
    for config_id, signs in _SIGN_DR_DESIGN[split].items():
        cls = _register_sign_config(config_id, signs)
        specs[config_id] = {
            "robot": cls.__name__,
            "joint_signs": list(signs),
        }
    return specs


SIGN_DR_TRAIN_CONFIG_SPECS = _build_sign_dr_specs("train")
SIGN_DR_OOD_CONFIG_SPECS = _build_sign_dr_specs("ood")


def get_validation_config_specs(config_set: str) -> dict[str, dict]:
    canonical = SIGN_DR_TRAIN_CONFIG_SPECS["sign_train_00"]
    config_sets = {
        "legacy": CONFIG_SPECS,
        "sign-train": {
            "cfg0": canonical,
            **{
                config_id: spec
                for config_id, spec in SIGN_DR_TRAIN_CONFIG_SPECS.items()
                if config_id != "sign_train_00"
            },
        },
        "sign-ood": {"cfg0": canonical, **SIGN_DR_OOD_CONFIG_SPECS},
    }
    try:
        return config_sets[config_set]
    except KeyError as error:
        raise ValueError(
            f"Unknown validation config set {config_set!r}; "
            f"expected one of {tuple(config_sets)}"
        ) from error
