import numpy as np
import cv2 as cv
import glob

import pickle






################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (9,6)
frameSize = (3072,4096)



# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


images = glob.glob("C:/Users/ITEMS/Desktop/strabismus-detection/kalibrasyon/*.jpg")
count = 0
for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    print(gray)
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        count += 1
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        
        # Resize the image to 1280x720 before showing
        resized_img = cv.resize(img, (1280, 720))
        cv.imshow('img', resized_img)
        cv.waitKey(1000)
print(count)


cv.destroyAllWindows()




############## CALIBRATION #######################################################
#print(objpoints)
#print(imgpoints)
#print(frameSize)

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)


############## UNDISTORTION #####################################################

img = cv.imread('result.jpg')
h,  w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))



# Undistort
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.jpg', dst)



# Undistort with Remapping
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w,h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.jpg', dst)




# Reprojection Error
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )
# Kalibrasyon sonuçlarını içeren bir sözlük oluşturun
calibration_data = {
    "cameraMatrix": cameraMatrix,
    "dist": dist,
    "rvecs": rvecs,
    "tvecs": tvecs,
    "newCameraMatrix": newCameraMatrix,
    "roi": roi
}
# Sözlüğü bir pickle dosyasına kaydedin
with open("calibration.pickle", "wb") as f:
    pickle.dump(calibration_data, f)

print("Kalibrasyon sonuçları 'calibration.pickle' dosyasına kaydedildi.")
