# ----------------------------------------------------------------------
# KALİBRASYON SINIFI (değiştirilmedi)
# ----------------------------------------------------------------------
import os
import pickle
import cv2
from tkinter import messagebox

class Calibrator:
    def calibrate(self, image_path, current_dir):
        calibration_folder = os.path.join(current_dir, "kalibrasyon")
        pickle_file = os.path.join(calibration_folder, "calibration.pickle")
        if not os.path.exists(pickle_file):
            messagebox.showwarning("Uyarı", "Kalibrasyon pickle dosyası bulunamadı!")
            return None
        try:
            with open(pickle_file, "rb") as f:
                calib_data = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Hata", f"Kalibrasyon dosyası okunamadı: {e}")
            return None
        camera_matrix = calib_data.get("cameraMatrix")
        dist_coeffs = calib_data.get("dist")
        new_camera_matrix = calib_data.get("newCameraMatrix")
        roi = calib_data.get("roi")
        if camera_matrix is None or dist_coeffs is None:
            messagebox.showerror("Hata", "Geçersiz kalibrasyon verisi!")
            return None

        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Hata", f"Görüntü yüklenemedi: {image_path}")
            return None

        h, w = image.shape[:2]
        if new_camera_matrix is None or roi is None:
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
        if roi is not None and len(roi) == 4:
            x, y, w_roi, h_roi = roi
            undistorted_image = undistorted_image[y:y+h_roi, x:x+w_roi]
        return undistorted_image

