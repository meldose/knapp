import numpy as np
import cv2
import scipy

def eye_to_hand_calib(target_poses, robot_poses):
    """
    target_poses (target2cam) are provided as list of tuples of (rvec, tvec).
    robot_poses (gripper2base) are provided as list of lists [x, y, z, w, p, r]
    """

    # 1. Collect target poses
    R_target2cam = []
    t_target2cam = []
    for pose in target_poses:
        rvec, tvec = pose
        R_target2cam.append(rvec)
        t_target2cam.append(tvec)

    # 2. Collect robot poses
    R_gripper2base = []
    t_gripper2base = []
    for pose in robot_poses:
        # gripper2base
        t = pose[0:3]
        # gripper2base
        wpr = pose[3:]
        R = scipy.spatial.transform.Rotation.from_euler(
            "XYZ", wpr, degrees=True
        ).as_matrix()

        R_gripper2base.append(R)
        t_gripper2base.append(t)

    # 3. Transform from gripper2base to base2gripper
    R_base2gripper = []
    t_base2gripper = []
    for R_g2b, t_g2b in zip(R_gripper2base, t_gripper2base):
        R_b2g = R_g2b.T
        t_b2g = np.matmul(-R_b2g, t_g2b)

        R_base2gripper.append(R_b2g)
        t_base2gripper.append(t_b2g)

    # 4. Call calibration
    # R_calib, t_calib = cv2.calibrateHandEye(
    #     R_gripper2base=R_base2gripper,
    #     t_gripper2base=t_base2gripper,
    #     R_target2cam=R_target2cam,
    #     t_target2cam=t_target2cam,
    # )

    R_calib, t_calib = cv2.calibrateHandEye(
        R_gripper2base=R_target2cam,
        t_gripper2base=t_target2cam,
        R_target2cam=R_base2gripper,
        t_target2cam=t_base2gripper,
    )
    print("R_cam2base: \n", R_calib)
    print("t_cam2base: \n", t_calib)

    H_base_camera = np.r_[np.c_[R, t_calib.flatten()], [[0, 0, 0, 1]]]
    return H_base_camera