import cv2 as cv
import numpy as np




def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, pts = cv.findChessboardCorners(gray, board_pattern)
        if ret:
            img_points.append(pts)
    assert len(img_points) > 0, "체커보드가 감지된 프레임이 없습니다다"

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)

    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)


# --- 설정 ---
video_name = "chessboard.avi"
output_name = "AR.avi"
board_pattern = (10, 7)
board_cellsize = 25.0
fourcc = cv.VideoWriter_fourcc(*'XVID')




video = cv.VideoCapture(video_name)
video_images = []

tri_lower = board_cellsize * np.array([[2, 1, 0], [6, 1, 0], [6, 5, 0]])
tri_upper = board_cellsize * np.array([4, 2,-1])

obj_points = board_cellsize * np.array([[c,r,0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

ret, sample_img = video.read()
frame_height, frame_width = sample_img.shape[:2]
video.set(cv.CAP_PROP_POS_FRAMES, 0)

out = cv.VideoWriter(output_name, fourcc, 30.0, (frame_width, frame_height))

while True :
    vaild, img = video.read()
    if not vaild:
        break

    video_images.append(img)


# --- 카메라 캘리브레이션 ---
if len(video_images) > 0:
    ret, K, dist_coeffs, rvecs, tvecs = calib_camera_from_chessboard(video_images, board_pattern, board_cellsize)

# print(f"{K}")
# print(f"{dist_coeffs}")

# K = np.array([[1.09, 0, 6.32], [0, 1.09, 3.39], [0,0,1]])
# dist_coeffs = np.array([0.04,0.22,0,0,-0.92])


video = cv.VideoCapture(video_name)


while True:

    vaild, img = video.read()
    if not vaild:
        break

    complete, img_points = cv.findChessboardCorners(img, board_pattern)

    if complete:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeffs)

        line_lower, _ = cv.projectPoints(tri_lower, rvec, tvec, K, dist_coeffs)
        point_upper, _ = cv.projectPoints(tri_upper, rvec, tvec, K, dist_coeffs)
        # 밑면 그리기기
        cv.polylines(img, [np.int32(line_lower)], True, (255, 0, 0), 2)

        pt_upper = tuple(np.int32(point_upper[0].flatten()))

        for pt in line_lower :
            pt_lower = tuple(np.int32(pt.flatten()))
            cv.line(img, pt_lower, pt_upper, (0,255,0), 2)


        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'

        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    out.write(img)


video.release()
out.release()
cv.destroyAllWindows()

