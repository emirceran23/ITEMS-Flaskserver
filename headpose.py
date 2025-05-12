#!/usr/bin/env python
# headpose.py
"""
headpose.py
-----------
Bu modül, yüzün head pose (baş pozisyonu) analizi ve roll düzeltmesi (frontalizasyon)
işlemlerini gerçekleştiren HeadPoseEstimator sınıfını içerir.
"""

import cv2
import numpy as np
import math

def normalize_angle(angle):
    """
    Verilen açıyı -90 ile +90 aralığına normalize eder.
    
    Örnek:
      175° -> 175 - 180 = -5°
      -170° -> -170 + 180 = 10°
    """
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle

class HeadPoseEstimator:
    """
    HeadPoseEstimator sınıfı, yüz landmark'larını kullanarak baş pozisyonunu (pitch, yaw, roll)
    tahmin eder ve isteğe bağlı olarak roll düzeltmesi (frontalizasyon) uygular.
    """
    def __init__(self):
        # Gerekirse burada parametreler ya da ön hesaplamalar yapılabilir.
        pass

    def estimate_head_pose_full(self, face_result, image):
        """
        Yüz landmark'larını kullanarak head pose hesaplaması yapar.
        Gerekli landmark indeksleri: [1, 152, 33, 263, 61, 291].
        
        Parametreler:
          face_result: Yüz analizi sonucu (dict veya liste formatında landmark bilgileri).
          image: Giriş görüntüsü (BGR formatında).
        
        Return:
          (pitch, yaw, roll, rotation_vector, R, camera_matrix) ya da None.
        """
        # Landmark'ları elde etme
        if isinstance(face_result, dict):
            landmarks = face_result.get('landmarks', None)
            if landmarks is None:
                landmarks = getattr(face_result, 'landmarks', None)
        elif isinstance(face_result, list):
            landmarks = face_result
        else:
            raise ValueError("Beklenmeyen yüz sonucu formatı.")
        
        if landmarks is None:
            raise ValueError("Yüz landmark'ları bulunamadı.")
        
        image_h, image_w = image.shape[:2]
        
        def convert_point(pt):
            """
            Noktayı normalize eder. Eğer koordinatlar 0-1 aralığındaysa, piksele çevirir.
            """
            if isinstance(pt, (list, tuple)):
                x, y = pt[0], pt[1]
                if x <= 1.0 and y <= 1.0:
                    return [x * image_w, y * image_h]
                else:
                    return [x, y]
            elif hasattr(pt, 'x') and hasattr(pt, 'y'):
                x, y = pt.x, pt.y
                if x <= 1.0 and y <= 1.0:
                    return [x * image_w, y * image_h]
                else:
                    return [x, y]
            else:
                raise ValueError("Beklenmeyen nokta formatı: {}".format(pt))
        
        indices = [1, 152, 33, 263, 61, 291]
        image_points = []
        for idx in indices:
            try:
                point = convert_point(landmarks[idx])
            except IndexError:
                raise IndexError("Beklenen landmark indeksi {} mevcut değil.".format(idx))
            image_points.append(point)
        image_points = np.array(image_points, dtype="double")
        
        # 3D model noktaları (ortalama yüz geometrisi)
        model_points = np.array([
            [0.0, 0.0, 0.0],          # Nose tip
            [0.0, -330.0, -65.0],     # Chin
            [-225.0, 170.0, -135.0],  # Left eye left corner
            [225.0, 170.0, -135.0],   # Right eye right corner
            [-150.0, -150.0, -125.0], # Left Mouth corner
            [150.0, -150.0, -125.0]   # Right mouth corner
        ])
        
        # Kamera parametreleri
        focal_length = image_w
        center = (image_w / 2, image_h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        dist_coeffs = np.zeros((4, 1))
        
        # solvePnP ile poz tahmini
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return None
        
        # Rodrigues dönüşümü ile dönüş matrisini elde etme
        R, _ = cv2.Rodrigues(rotation_vector)
        proj_matrix = np.hstack((R, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        pitch, yaw, roll = [angle[0] for angle in euler_angles]
        return pitch, yaw, roll, rotation_vector, R, camera_matrix

    def estimate_head_pose(self, face_result, image):
        """
        Sadece pitch, yaw ve roll değerlerini döndüren kısa arayüz.
        """
        res = self.estimate_head_pose_full(face_result, image)
        if res is None:
            return None
        pitch, yaw, roll, _, _, _ = res
        return pitch, yaw, roll

    def frontalize_face(self, image, left_eye_bbox, right_eye_bbox):
        """
        Göz bounding box'larından göz merkezlerini hesaplayarak roll açısına göre görüntüyü döndürür.
        Dönüşüm matrisinde +angle kullanılarak doğru yönde döndürme sağlanır.
        
        Parametreler:
          image: Giriş görüntüsü.
          left_eye_bbox: Sol göz için (x1, y1, x2, y2) koordinatları.
          right_eye_bbox: Sağ göz için (x1, y1, x2, y2) koordinatları.
        
        Return:
          (rotated_image, dönüşüm matrisi)
        """
        # Göz merkezlerini hesapla
        left_center = ((left_eye_bbox[0] + left_eye_bbox[2]) // 2,
                       (left_eye_bbox[1] + left_eye_bbox[3]) // 2)
        right_center = ((right_eye_bbox[0] + right_eye_bbox[2]) // 2,
                        (right_eye_bbox[1] + right_eye_bbox[3]) // 2)
        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        angle = math.degrees(math.atan2(dy, dx))
        # Görüntüyü göz merkezine göre döndür
        eyes_center = ((left_center[0] + right_center[0]) // 2,
                       (left_center[1] + right_center[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return rotated, M

    def frontalize_face_roll_only(self, image, face_result, left_eye_bbox, right_eye_bbox):
        """
        Head pose'dan elde edilen roll değerine dayanarak yalnızca roll düzeltmesi yapar.
        Pitch ve yaw sabit kalır.
        
        Return:
          (rotated_image, dönüşüm matrisi)
        """
        rotated, M = self.frontalize_face(image, left_eye_bbox, right_eye_bbox)
        return rotated, M
