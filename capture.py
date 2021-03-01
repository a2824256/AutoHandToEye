import pyrealsense2 as rs
import numpy as np
import cv2
import json
import png
import cv2.aruco as aruco

pipeline = rs.pipeline()

config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)
depth_intrin = None
def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # 深度相关参数
    # depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    color_frame = aligned_frames.get_color_frame()
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    camera_parameters = {'fx': intr.fx, 'fy': intr.fy,
                         'ppx': intr.ppx, 'ppy': intr.ppy,
                         'height': intr.height, 'width': intr.width,
                         'depth_scale': profile.get_device().first_depth_sensor().get_depth_scale()
                         }
    intr_matrix = np.array([
        [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
    ])
    with open('./intrinsics.json', 'w') as fp:
        json.dump(camera_parameters, fp)
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    depth_image_8bit = cv2.convertScaleAbs(depth_image, alpha=0.03)
    pos=np.where(depth_image_8bit==0)
    depth_image_8bit[pos]=255
    color_image = np.asanyarray(color_frame.get_data())
    # d = aligned_depth_frame.get_distance(320, 240)
    return color_image, depth_image, intr_matrix, np.array(intr.coeffs)


if __name__ == "__main__":
    n=0
    while 1:
        rgb, depth, intr_matrix, intr_coeffs = get_aligned_images()
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejected_img_points = aruco.detectMarkers(rgb, aruco_dict, parameters=parameters,
                                                                cameraMatrix=intr_matrix, distCoeff=intr_coeffs)
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.045, intr_matrix,
                                                                   intr_coeffs)
        try:
            aruco.drawDetectedMarkers(rgb, corners)
            aruco.drawAxis(rgb, intr_matrix, intr_coeffs, rvec, tvec, 0.05)
            cv2.imshow('RGB image', rgb)
        except:
            cv2.imshow('RGB image', rgb)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            pipeline.stop()
            break
        elif key==ord('s'):
            n=n+1
            # 保存rgb图
            cv2.imwrite('./bd/rgb' + str(n)+'.jpg',rgb)
            # 保存16位深度图
            with open('./bd/rgb' + str(n) + "_d.jpg", 'wb') as f:
                writer = png.Writer(width=depth.shape[1], height=depth.shape[0],
                                    bitdepth=16, greyscale=True)
                zgray2list = depth.tolist()
                writer.write(f, zgray2list)

    cv2.destroyAllWindows()