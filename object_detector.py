import numpy as np
import cv2
import pyrealsense2 as rs
import time

class ObjectDetector:
    def __init__(self, calibration_matrix_path, min_distance=0.2, max_distance=1.5):
        self.T_cam_to_tcp = np.load(calibration_matrix_path)
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.profile = self.pipeline.start(cfg)
        self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.align = rs.align(rs.stream.color)
        self.latest_color_image = None
        self.min_distance = min_distance
        self.max_distance = max_distance

        for _ in range(5):
            self.pipeline.wait_for_frames()

    def get_detection(self, retry_count=5, delay_sec=0.5, visualize=True):
        for attempt in range(retry_count):
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            self.latest_color_image = color_image
            depth_image_raw = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.convertScaleAbs(depth_image_raw, alpha=0.03)

            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            dilated = cv2.dilate(edges, None, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contour = None

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 2000 or area > 20000:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter ** 2)
                if circularity < 0.5:
                    continue

                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                if hull_area == 0:
                    continue
                solidity = area / float(hull_area)
                if solidity < 0.85:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if aspect_ratio < 0.8 or aspect_ratio > 1.2:
                    continue

                depth_roi = depth_image_raw[y:y+h, x:x+w]
                depth_roi = depth_roi[depth_roi > 0]
                if len(depth_roi) == 0:
                    continue
                depth_value = np.median(depth_roi) * 0.001

                if not (self.min_distance <= depth_value <= self.max_distance):
                    continue

                mask = np.zeros(depth_image_raw.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                depth_inside = depth_image_raw[mask == 255]
                depth_inside = depth_inside[depth_inside > 0]
                if len(depth_inside) < 10 or np.std(depth_inside) > 30:
                    continue

                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

                angle = self.compute_orientation(contour)

                valid_contour = contour
                break

            if valid_contour is None:
                self._show_no_object(color_image, show=visualize)
                time.sleep(delay_sec)
                continue

            xyz_camera = rs.rs2_deproject_pixel_to_point(self.intrinsics, [center_x, center_y], depth_value)
            xyz_camera = np.array([*xyz_camera, 1], dtype=np.float32)
            position_tcp = self.T_cam_to_tcp @ xyz_camera

            if visualize:
                self._show_detection(color_image, valid_contour, (center_x, center_y), angle)

            return {
                "pixel": (center_x, center_y),
                "position_camera": xyz_camera[:3],
                "position_tcp": position_tcp[:3],
                "orientation_deg": angle,
                "color_image": color_image,
                "depth_frame": depth_frame,
                "depth_image": depth_colormap
            }

        print("[ObjectDetector] No object detected after retries.")
        return None

    def compute_orientation(self, contour):
        data_pts = np.array(contour, dtype=np.float64).reshape(-1, 2)
        mean, eigenvectors = cv2.PCACompute(data_pts, mean=np.empty((0)))
        angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
        return np.degrees(angle)

    def _show_no_object(self, image, show=True):
        if show and image is not None:
            img = image.copy()
            cv2.putText(img, "No object found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
            cv2.imshow("Detection", img)
            cv2.waitKey(1)

    def _show_detection(self, image, contour, center, angle):
        img = image.copy()
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        box = np.intp(box)
        cv2.drawContours(img, [box], 0, (255, 255, 0), 2)
        cx, cy = center
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(img, f"Angle: {angle:.1f}", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Object Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Detection", img)
        cv2.waitKey(1)

    def get_last_color_image(self):
        return self.latest_color_image

    def get_focal_length(self, average=False):
        return (self.intrinsics.fx + self.intrinsics.fy) / 2 if average else self.intrinsics.fx

    def release(self):
        self.pipeline.stop()
