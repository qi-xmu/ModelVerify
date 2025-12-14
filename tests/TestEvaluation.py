import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from base.datatype import Pose, PosesData
from base.evaluate import Evaluation


class TestEvaluation(unittest.TestCase):
    def test_ape(self):
        ref_poses = PosesData(
            t_us=np.array([0, 1e6]),
            rots=Rotation.from_rotvec([[0, 1, 0], [0, 1, 0]]),
            ps=np.array([[0, 0, 0], [1, 0, 0]]),
        )
        eva_poses = PosesData(
            t_us=np.array([0, 1e6]),
            rots=Rotation.from_rotvec([[0, 0, 1], [0, 1, 0]]),
            ps=np.array([[0, 0, 1], [0, 1, 0]]),
        )

        evaluator = Evaluation(ref_poses)
        evaluator.get_eval(eva_poses, "test")
        evaluator.print()


if __name__ == "__main__":
    unittest.main()
