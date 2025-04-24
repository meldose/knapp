import pyrealsense2 as rs
import numpy as np
import open3d as o3d

# 1. Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 2. Start streaming
pipeline.start(config)

# 3. Get camera intrinsics once
profile = pipeline.get_active_profile()
depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
intrinsics = depth_profile.get_intrinsics()

pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
    width=intrinsics.width,
    height=intrinsics.height,
    fx=intrinsics.fx,
    fy=intrinsics.fy,
    cx=intrinsics.ppx,
    cy=intrinsics.ppy,
)

try:
    while True:
        # 4. Wait for a frame
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # 5. Convert to numpy
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 6. Convert to Open3D format
        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(color_image)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            convert_rgb_to_intensity=False,
            depth_scale=1000.0,  # RealSense gives depth in mm
            depth_trunc=3.0
        )

        # 7. Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            pinhole_camera_intrinsic
        )
        pcd.transform([[1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, -1, 0],
                       [0, 0, 0, 1]])

        # 8. Visualize one frame (static view)
        o3d.visualization.draw_geometries([pcd])

        # To make this a *live viewer*, you'd need a custom Open3D visualizer loop

finally:
    pipeline.stop()
