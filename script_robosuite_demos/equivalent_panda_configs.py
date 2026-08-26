"""Kinematically equivalent Panda coordinate configurations.

Each configuration changes the sign convention of one joint coordinate in the
MJCF model.  Joint axis, range, initial position, state, and action must all be
transformed by the same sign.  The physical robot geometry is therefore
unchanged even though the robot model and action labels differ.
"""

from __future__ import annotations

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

