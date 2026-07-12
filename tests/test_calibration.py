from __future__ import annotations

from traffic_intel.calibration import Calibration


def test_identity_homography_projects_inside_calibrated_zone() -> None:
    calibration = Calibration(
        image_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        world_points=[[0, 0], [10, 0], [10, 10], [0, 10]],
        homography_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    assert calibration.world_from_image(5, 5) == (5.0, 5.0)
    assert calibration.world_from_image(100, 100) is None
    assert calibration.quality_grade == "EXCELLENT"
