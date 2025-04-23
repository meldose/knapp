import numpy as np # import numpy module
import cv2 # import cv2 module
from neurapy.robot import Robot
from object_detector import ObjectDetector # importing object_detector
from robot_controller import RobotController # importing robot_controller

def main():
    r = Robot()
    r.set_mode("Automatic")
    r.move_joint("New_capture")

    calibration_matrix = r"/home/hrg/Documents/package_detection/cam_to_tcp_transform.npy"
    detector = ObjectDetector(calibration_matrix)
    robot_control = RobotController()

    try:
        while True:
            # Get detection with visualization enabled
            detection = detector.get_detection(visualize=True)

            if detection is None:
                print("[DEBUG] No object detected.")
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            print("[DETECTED] Camera XYZ:", detection["position_camera"], "Angle deg:", detection["orientation_deg"])

            tcp_pose_current = robot_control.robot.get_tcp_pose()
            T_tcp_to_base = robot_control.pose_to_matrix(tcp_pose_current, gripper_offset_z=-0.103)

            pos_cam_hom = np.array([*detection["position_camera"], 1])
            print("[HOMOGENEOUS CAMERA POSITION]", pos_cam_hom)

            base_coords = T_tcp_to_base @ detector.T_cam_to_tcp @ pos_cam_hom
            print("[BASE COORDINATES]", base_coords)

            yaw_rad = np.deg2rad(detection["orientation_deg"])
            print("[YAW RADIANS]", yaw_rad)

            target_pose = [base_coords[0], base_coords[1], base_coords[2], 0, np.pi, yaw_rad]
            print("[TARGET POSE]", target_pose)

            yaw_deg = np.rad2deg(yaw_rad)
            print("[YAW DEGREES]", yaw_deg)

            # Visual marker already shown inside get_detection, but you can still highlight extra here
            img = detection["color_image"]
            cv2.circle(img, detection["pixel"], 5, (0, 255, 0), -1)
            cv2.imshow("Detection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            robot_control.check_gripper_status()

            robot_control.move_to_pose(target_pose, speed=0.06)
            print("STARTING THE ROBOT")
            robot_control.move_robot_based_on_angle(yaw_rad=yaw_rad)
            print("FINISHED ::::::::::::::::::::")

    except Exception as ex:
        print("[ERROR]", ex)
    finally:
        detector.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


