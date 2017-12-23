# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys

def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")

def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_black = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,0), fx=scale_factor,fy=scale_factor)
    ref_color = cv2.resize(cv2.imread("images/pattern001.jpg", cv2.IMREAD_COLOR), (0,0), fx=scale_factor,fy=scale_factor)
    ref_avg   = (ref_white + ref_black) / 2.0
    ref_on    = ref_avg + 0.05 # a threshold for ON pixels
    ref_off   = ref_avg - 0.05 # add a small buffer region
    h,w = ref_white.shape
    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h,w), dtype=np.uint16)

    # analyze the binary patterns from the camera15
    for i in range(0,15):
        # read the file
        patt_gray = cv2.resize(cv2.imread("images/pattern%03d.jpg"%(i+2),cv2.IMREAD_GRAYSCALE)/255.0, (0,0), fx=scale_factor,fy=scale_factor)
        # mask where the pixels are ON
        on_mask = (patt_gray > ref_on) & proj_mask

        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        tmp = bit_code*on_mask
        scan_bits = scan_bits|tmp
        # TODO: populate scan_bits by putting the bit_code according to on_mask

    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl","r") as f:
        binary_codes_ids_codebook = pickle.load(f)
    camera_points = []
    projector_points = []
    corr_img = np.zeros((h,w,3), dtype=np.uint8)
    for x in range(w):
        for y in range(h):
            if not proj_mask[y,x]:
                continue # no projection here
            if scan_bits[y,x] not in binary_codes_ids_codebook:
                continue # bad binary code
            x_p, y_p = binary_codes_ids_codebook.get(scan_bits[y][x])
            if x_p >= 1279 or y_p >= 799: # filter
                continue
            ref_color[y][x]
            camera_points.append((x/2.0, y/2.0))
            projector_points.append((x_p, y_p))
            y_p = 255*y_p/(x_p+y_p)
            x_p = 255*x_p/(x_p+y_p)
            corr_img[y,x,:] = np.array([0,y_p,x_p])
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points

            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2
    output_name = sys.argv[1] + "correspondence.jpg"
    cv2.imwrite(output_name, corr_img)
    # load the prepared stereo calibration between projector and camera
    with open("stereo_calibration.pckl","r") as f:
        d = pickle.load(f)
        camera_K    = d['camera_K']
        camera_d    = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    cam_pts = np.array(camera_points, np.float32)
    cam_pts = cam_pts.reshape(-1, 1, 2)
    proj_pts = np.array(projector_points, np.float32)
    proj_pts = proj_pts.reshape(-1, 1, 2)
    camera_pointsOut = cv2.undistortPoints(cam_pts, camera_K, camera_d)
    proj_pointsOut = cv2.undistortPoints(proj_pts, projector_K, projector_d)
    P_rt = np.append(projector_R, projector_t, axis=1)
    P_rt = np.vstack([P_rt, [0.,0.,0.,1.]])
    I = np.identity(4)
    P1 = I[:3]
    P2 = P_rt[:3]
    P1 = np.float32(P1)
    P2 = np.float32(P2)
    proj_pointsOut = np.float32(proj_pointsOut)
    camera_pointsOut = np.float32(camera_pointsOut)

    out_homo = cv2.triangulatePoints(P1, P2, camera_pointsOut, proj_pointsOut)
    out_homo=np.transpose(out_homo)
    points_3d = cv2.convertPointsFromHomogeneous(out_homo)
    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d
    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d
    # TODO: use cv2.triangulatePoints to triangulate the normalized points
    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    # TODO: name the resulted 3D points as "points_3d"
    mask = (points_3d[:,:,2] > 200) & (points_3d[:,:,2] < 1400)
    points_3d = points_3d[mask]
    pts_clrd = []
    for i in range(points_3d.shape[0]):

        x = int(points_3d[i][0]*2)
        y = int(camera_points[i][1]*2)
        pts_clrd.append(ref_color[y][x])
    pts_3dClrd = np.append(points_3d, pts_clrd, axis = 1)

    # print(points_3d.shape)
    output_name = sys.argv[1] + "output_rgb.xyz"
    with open(output_name,"w") as f:
        for p in pts_3dClrd:
            f.write("%d %d %d %d %d %d\n"%(p[0],p[1],p[2],p[3],p[4],p[5]))
    return points_3d

def write_3d_points(points_3d):
    # ===== DO NOT CHANGE THIS FUNCTION =====
    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name,"w") as f:
        for p in points_3d:
            f.write("%d %d %d\n"%(p[0],p[1],p[2]))

    return points_3d

if __name__ == '__main__':
    # ===== DO NOT CHANGE THIS FUNCTION =====
    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
