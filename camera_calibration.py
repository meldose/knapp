import numpy as np # import numpy module
import cv2 # import cv2 module 
import os # import os module
import json # immport json module
import time # import time module
from datetime import datetime # import datetime module
import scipy

class CalibrationSystem: # defining the class
    def __init__(self, checkerboard_size=(9, 6), square_size=0.025, save_dir="calibration_data"): # initializing the class
        """
        Initialize calibration system
        
        Args:
            checkerboard_size: Number of inner corners (width, height)
            square_size: Size of each square in meters
            save_dir: Directory to save calibration data
        """
        self.checkerboard_size = checkerboard_size # defining the checkerboard size
        self.square_size = square_size # defining the square size
        self.save_dir = save_dir # defining the save directory
        
        # Create directories if they don't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Generate object points (3D points of checkerboard corners in checkerboard coordinate system)
        self.objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size
        
        # Camera matrix and distortion coefficients (need to be set from prior camera calibration)
        self.camera_matrix = None
        self.dist_coeffs = None
        
    def set_camera_params(self, camera_matrix, dist_coeffs): # defining the function for setting camera parameters
        """Set camera intrinsic parameters from prior calibration"""
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
    
    def capture_pose(self, image, robot_pose, capture_id=None): # defining the function for capturing pose
        """
        Capture checkerboard and robot pose
        
        Args:
            image: Camera image
            robot_pose: Dictionary with 'position' [x,y,z] and 'rotation_matrix' (3x3)
            capture_id: Optional identifier for this pose
        
        Returns:
            success: Whether checkerboard was found
            data: Captured pose data if successful
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        image="/home/midhun.eldose/Desktop/package/checkerboard_9x6_25mm.png"
        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if found:
            # Refine corner detection
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # Calculate pose of the checkerboard
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                ret, rvec, tvec = cv2.solvePnP(self.objp, corners, self.camera_matrix, self.dist_coeffs)
                
                # Convert rotation vector to rotation matrix
                R_target2cam, _ = cv2.Rodrigues(rvec)
                t_target2cam = tvec.reshape(-1)
                
                # Get timestamp or use provided ID
                timestamp = capture_id if capture_id is not None else datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                
                # Create data structure
                data = {
                    "id": timestamp,
                    "target2cam": {
                        "rotation_matrix": R_target2cam.tolist(),
                        "translation": t_target2cam.tolist()
                    },
                    "gripper2base": {
                        "rotation_matrix": robot_pose["rotation_matrix"],
                        "translation": robot_pose["position"]
                    }
                }
                
                # Save data to file
                filename = os.path.join(self.save_dir, f"pose_{timestamp}.json")
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=4)
                
                # Optionally save the image
                cv2.imwrite(os.path.join(self.save_dir, f"image_{timestamp}.jpg"), image)
                
                # Draw and display corners for visualization
                vis_img = cv2.drawChessboardCorners(image.copy(), self.checkerboard_size, corners, found)
                
                return True, data, vis_img
            else:
                print("Camera intrinsics not set. Please run set_camera_params first.")
                return False, None, image
        else:
            return False, None, image
    
    def load_calibration_data(self): # loading the calibration data
        """
        Load all saved calibration poses from the save directory
        
        Returns:
            R_gripper2base: List of rotation matrices
            t_gripper2base: List of translation vectors
            R_target2cam: List of rotation matrices
            t_target2cam: List of translation vectors
        """
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []
        
        # Get all JSON files
        json_files = [f for f in os.listdir(self.save_dir) if f.endswith('.json') and f.startswith('pose_')]
        print(json_files)
        
        for json_file in json_files:
            with open(os.path.join(self.save_dir, json_file), 'r') as f:
                data = json.load(f)
            
            # Extract data
            R_gripper2base.append(np.array(data["gripper2base"]["rotation_matrix"]))
            t_gripper2base.append(np.array(data["gripper2base"]["translation"]))
            R_target2cam.append(np.array(data["target2cam"]["rotation_matrix"]))
            t_target2cam.append(np.array(data["target2cam"]["translation"]))
        
        print(f"Loaded {len(json_files)} calibration poses")
        return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


########### EYE TO HAND CALIBRATION ##########################
    #def eye_to_hand_calib(self, R_gripper2base, target_poses, robot_poses,t_gripper2base, R_target2cam, t_target2cam, eye_to_hand=True): # calibrate eye-hand
    def eye_to_hand_calib(target_poses, robot_poses,eye_to_hand=True):
        """
        target_poses (target2cam) are provided as list of tuples of (rvec, tvec).
        robot_poses (gripper2base) are provided as list of lists [x, y, z, w, p, r]
        """
        if eye_to_hand:
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
                R_b2g = -R_g2b.T
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

            result = {
                "eye_to_hand": eye_to_hand,
                "calibration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "rotation_matrix": R.tolist(),
                "translation_vector": t.tolist(),
                "num_poses_used": len(R_gripper2base)
            }

            with open(os.path.join(self.save_dir, "calibration_result.json"), 'w') as f:
                    json.dump(result, f, indent=4)
                    

            H_base_camera = np.r_[np.c_[R, t_calib.flatten()], [[0, 0, 0, 1]]]
            return H_base_camera


        

        #     if eye_to_hand:
        #         # change coordinates from gripper2base to base2gripper
        #         R_base2gripper, t_base2gripper = [], []
        #         for R, t in zip(R_gripper2base, t_gripper2base):
        #             R_b2g = R.T
        #             t_b2g = -R_b2g @ t
        #             R_base2gripper.append(R_b2g)
        #             t_base2gripper.append(t_b2g)
                
        #         # change parameters values
        #         R_gripper2base = R_base2gripper
        #         t_gripper2base = t_base2gripper
                
        #     # calibrate
        #         R, t = cv2.calibrateHandEye(
        #             R_gripper2base=R_gripper2base,
        #             t_gripper2base=t_gripper2base,
        #             R_target2cam=R_target2cam,
        #             t_target2cam=t_target2cam,
        #     )
        #         print("R_cam2base: \n", R_calib)
        #         print("t_cam2base: \n", t_calib)

        #         H_base_camera = np.r_[np.c_[R, t_calib.flatten()], [[0, 0, 0, 1]]]
        #         return H_base_camera
        # # Save calibration result
        #     result = {
        #         "eye_to_hand": eye_to_hand,
        #         "calibration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #         "rotation_matrix": R.tolist(),
        #         "translation_vector": t.tolist(),
        #         "num_poses_used": len(R_gripper2base)
        #     }
            
        #     with open(os.path.join(self.save_dir, "calibration_result.json"), 'w') as f:
        #             json.dump(result, f, indent=4)
                    
        #     return R, t

    def calculate_reprojection_error(self, R_cam2gripper, t_cam2gripper, R_gripper2base, t_gripper2base,R_target2cam, t_target2cam):
        """
        Calculate reprojection error of the calibration
        
        Returns:
            errors: List of transformation errors for each pose
        """
        errors = []
        
        for i in range(len(R_gripper2base)):
            # Calculate AX = XB equation error
            R_cam2base = R_gripper2base[i] @ R_cam2gripper
            t_cam2base = R_gripper2base[i] @ t_cam2gripper + t_gripper2base[i]
            
            R_cam2target_computed = R_cam2base @ np.linalg.inv(R_target2cam[i])
            t_cam2target_computed = t_cam2base - R_cam2target_computed @ t_target2cam[i]
            
            # Calculate error as Frobenius norm of difference
            R_error = np.linalg.norm(R_cam2target_computed - np.eye(3), 'fro')
            t_error = np.linalg.norm(t_cam2target_computed)
            
            errors.append({"rotation_error": R_error, "translation_error": t_error})
            
        return errors
    
if __name__ == "__main__": # define main
    # Example usage
    calib = CalibrationSystem() # set up the calibration system
    

    calib.eye_to_hand_calib() # calling function for calibration
    # calib.load_calibration_data() # calling function for loading data
    # calib.calculate_reprojection_error() # calling function for calculating reprojection error