from gym.envs.registration import register
from .CamCalibrEnv import CamCalibrEnv, CamCalibrEnv_seq, imuCalibrEnv_seq, CamImuCalibrEnv_seq

register(
    id='CamCalibr-v1',
    entry_point='calibration_env:CamCalibrEnv'
)

register(
    id='CamCalibr-v2',
    entry_point='calibration_env:CamCalibrEnv_seq'
)

register(
    id='imuCalibr-v1',
    entry_point='calibration_env:imuCalibrEnv_seq'
)

register(
    id='camimuCalibr-v1',
    entry_point='calibration_env:CamImuCalibrEnv_seq'
)

