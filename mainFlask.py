#!/usr/bin/env python
# main.py
"""
main.py
--------
Bu dosya, genişletilmiş Tkinter arayüzü ile kullanıcıya:
  - Mevcut klasörü değiştirebilme,
  - Seçilen klasördeki resimleri listeleyebilme,
  - Seçilen resim önizlemesini görüntüleme,
  - "Analiz Et" ve "Kalibre Et" butonlarıyla işlemleri başlatma imkanı sunar.
  
Ayrıca, aynı dizinde yer alan Ultralytics YOLO v11 semantic segmentasyon modelini (best.pt)
kullanarak, iris, pupil ve glare (parlak yansıma) tespitlerini gerçekleştirir.
  
Kod, geleneksel yöntemleri (IrisLandmark/IrisAnalyzer) yorum satırına alarak artık tüm
hesaplamaları YOLO segmentasyonundan almaktadır.
  
"""

import os
import cv2
import numpy as np
import copy
import math
import pickle


from PIL import Image, ImageTk, ExifTags
import matplotlib.pyplot as plt
from calibratior import Calibrator
from matplotlib.patches import Circle, Wedge

import torch
import numpy as np
from PIL import Image as PILImage
# Ultralytics YOLO modülü
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics YOLO modülü bulunamadı! Lütfen 'pip install ultralytics' komutuyla yükleyiniz.")

from face_mesh.face_mesh import FaceMesh
from iris_landmark.iris_landmark import IrisLandmark
from headpose import HeadPoseEstimator, normalize_angle
from iris import IrisAnalyzer, MAX_ANGLE_PER_RADIUS
try:
    from torchvision import transforms
except ImportError:
    
    print("Torchvision modülü bulunamadı! Lütfen 'pip install torchvision' komutuyla yükleyiniz.")
from pdf_report_generator import PDFReportGenerator

from io import BytesIO
import math

FRONTALIZATION_THRESHOLD = 1.0
MIN_GLARE_RATIO = 0.02  # Iris alanının en az %2'si
MAX_GLARE_RATIO = 0.15  # En fazla %15'i

MM_PER_DEGREE = 0.2  # Örnek: 1 derece = 0.2 mm kayma









plt.ion()

# YOLO renk ve sınıf tanımlamaları
CLASS_COLORS = {
    0: (0, 0, 255),    # Iris → Kırmızı
    1: (0, 255, 0),    # Reflection → Yeşil
    2: (255, 0, 0)     # Pupil → Mavi
}
CLASS_NAMES = {
    0: "Iris",
    1: "Pupil",
    2: "Reflection"
}
mapping = {
    0: "Iris",
    1: "Pupil",
    2: "Reflection"
}

def create_mask_for_polygon(shape, polygon_pts):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_pts.astype(np.int32)], 255)
    return mask

def create_mask_for_circle(shape, center, radius):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(center[0]), int(center[1])), int(radius), 255, -1)
    return mask

def compute_polygon_center(pts):
    M = cv2.moments(pts)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(pts)
        cx = x + w // 2
        cy = y + h // 2
    return cx, cy
def compute_interPupilDistance_pinholes(left_pupil_center, right_pupil_center, sensor_width_mm, image_width_pixels, focal_length_mm, iris_depth_mm):
    """
    Pinhole kamera modeli kullanılarak, resimde ölçülen pupil mesafesi ve iris depth
    değerini esas alarak gerçek interpupiller mesafeyi (mm cinsinden) hesaplar.

    Parametreler:
      - left_pupil_center: Sol pupil merkezinin (x, y) koordinatları (piksel cinsinden).
      - right_pupil_center: Sağ pupil merkezinin (x, y) koordinatları (piksel cinsinden).
      - sensor_width_mm: Sensörün fiziksel genişliği (mm cinsinden).
      - image_width_pixels: Resmin genişliği (piksel cinsinden).
      - focal_length_mm: Kameranın fiziksel odak uzaklığı (mm cinsinden).
      - iris_depth_mm: Kamera ile iris arasındaki gerçek mesafe (mm cinsinden).

    Hesaplama:
      1. Öncelikle, pupil mesafesini piksel cinsinden hesaplayın.
      2. Piksel başına düşen fiziksel boyutu (mm/piksel) bulun:
            pixel_size = sensor_width_mm / image_width_pixels
      3. Resimdeki pupil mesafesinin sensör üzerindeki karşılığı:
            measured_distance_mm = pupil_distance_pixels * pixel_size
      4. Pinhole modeline göre gerçek interpupiller mesafe:
            real_pupil_distance_mm = (measured_distance_mm * iris_depth_mm) / focal_length_mm

    Dönüş:
      - real_pupil_distance_mm: Gerçek interpupiller mesafe (mm cinsinden).
    """
    # 1. Piksel cinsinden pupil mesafesi:
    dx = right_pupil_center[0] - left_pupil_center[0]
    dy = right_pupil_center[1] - left_pupil_center[1]
    pupil_distance_pixels = math.sqrt(dx**2 + dy**2)

    # 2. Piksel başına düşen fiziksel boyut (mm/piksel):
    pixel_size = sensor_width_mm / image_width_pixels

    # 3. Resimdeki pupil mesafesinin sensör üzerindeki karşılığı (mm):
    measured_distance_mm = pupil_distance_pixels * pixel_size

    # 4. Pinhole kamera modeline göre gerçek interpupiller mesafe (mm):
    #    Formül: object_size = (image_size * object_distance) / focal_length
    real_pupil_distance_mm = (measured_distance_mm * iris_depth_mm) / focal_length_mm

    print(f"Pupil Distance (Resimde, mm cinsinden): {measured_distance_mm:.2f} mm")
    print(f"Gerçek Pupil Mesafesi (Pinhole Modeline Göre): {real_pupil_distance_mm:.2f} mm")
    
    return real_pupil_distance_mm


    
    

def save_fig_to_file(fig, filename):
    """
    Verilen matplotlib figürünü belirtilen dosya adına PNG formatında kaydeder.
    """
    fig.savefig(filename, format="PNG", bbox_inches="tight")

    return filename


def ensure_temp_folder():
    """
    Geçici dosyaların kaydedileceği klasörün (temp_reports) varlığını kontrol eder, yoksa oluşturur.
    """
    folder = "temp_reports"
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder




def validate_glare(glare_mask, iris_area):
    """
    Verilen glare maskesinin alanını iris alanıyla karşılaştırır.
    Eğer oran belirlenen eşik aralığındaysa True döner.
    """
    glare_area = cv2.contourArea(glare_mask)
    ratio = glare_area / iris_area
    return MIN_GLARE_RATIO < ratio < MAX_GLARE_RATIO


def calculate_deviation(results,glare_center, pupil_center, calibration_factor=1.0):
    """
    Glare ve pupil merkezleri arasındaki kaymayı mm cinsinden hesaplar,
    sonrasında açı (derece) cinsinden sapma hesaplar.
    """
    dx_mm = (glare_center[0] - pupil_center[0]) * calibration_factor
    dy_mm = (glare_center[1] - pupil_center[1]) * calibration_factor
    distance_mm = math.sqrt(dx_mm**2 + dy_mm**2)
    deviation_deg = math.degrees(math.atan(distance_mm / results.get("PUPIL_DISTANCE")))
    return deviation_deg


def adjust_for_headpose(deviation, head_pose):
    """
    Ham sapma (deviation) değerini, başın pitch ve yaw açılarını dikkate alarak düzeltir.
    Parametreler:
      - deviation: Ham sapma değeri (örneğin, mm cinsinden hesaplanan sapma veya açı değeri)
      - head_pose: (pitch, yaw, roll) şeklinde bir tuple
    Döndürür:
      - Baş pozuna göre düzeltilmiş sapma değeri
    """
    # head_pose bir tuple olarak (pitch, yaw, roll) döndürüyor
    pitch, yaw, roll = head_pose
    # Yatay (yaw) ve düşey (pitch) düzeltme faktörleri: cosinus fonksiyonu kullanılarak hesaplanıyor.
    yaw_factor = math.cos(math.radians(yaw))
    pitch_factor = math.cos(math.radians(pitch))
    adjusted_deviation = deviation * math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))
    return adjusted_deviation


def draw_clinical_guidelines(
    orig_image,
    image, 
    ref_center=None, 
    focal_length=None, 
    calibration_data=None,
    iris_depth_mm=None,
    mm_range=(-5, 6),
    line_color=(0, 255, 0),
    thickness=1,
    font_scale=0.35
):
    """
    Draw horizontal and vertical clinical guidelines and mm tick marks
    around a reference center (e.g., the pupil center).
    """

    bold_thickness = 2
    if image is None or image.size == 0:
        print("draw_clinical_guidelines: Invalid or empty image!")
        return image

    # -- Original vs displayed image dimensions --
    ho, hw = orig_image.shape[:2]  # original image
    h, w = image.shape[:2]         # displayed/processed image
    pixel_proportion = float(w) / hw if hw else 1.0

    # -- Reference center --
    if ref_center is None:
        cx, cy = w // 2, h // 2
    else:
        cx, cy = ref_center

    # -- Main cross lines --
    cv2.line(image, (0, cy), (w, cy), line_color, thickness)
    cv2.line(image, (cx, 0), (cx, h), line_color, thickness)

    # -- Center highlight --
    cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

    # -- Pixel/mm conversion --
    PIXELS_PER_MM = compute_pixels_per_mm(focal_length, calibration_data)
    if PIXELS_PER_MM == 0:
        PIXELS_PER_MM = 1.0
    PIXELS_PER_MM = 1.0 / (PIXELS_PER_MM * pixel_proportion)

    # -- Edges in pixels --
    left_px, right_px = cx, (w - cx)
    top_px,  bottom_px = cy, (h - cy)

    # -- Convert to mm (basic) --
    left_mm   = left_px   * PIXELS_PER_MM
    right_mm  = right_px  * PIXELS_PER_MM
    top_mm    = top_px    * PIXELS_PER_MM
    bottom_mm = bottom_px * PIXELS_PER_MM

    # -- Pin-hole model if focal_length & iris_depth_mm are given --
    if focal_length and iris_depth_mm:
        left_eff   = (left_mm   * iris_depth_mm) / focal_length
        right_eff  = (right_mm  * iris_depth_mm) / focal_length
        top_eff    = (top_mm    * iris_depth_mm) / focal_length
        bottom_eff = (bottom_mm * iris_depth_mm) / focal_length
    else:
        left_eff, right_eff, top_eff, bottom_eff = left_mm, right_mm, top_mm, bottom_mm

    max_eff_mm = int(min(left_eff, right_eff, top_eff, bottom_eff))

    # -- Merge the user mm_range with the feasible mm range from the image size --
    start_mm = max(mm_range[0], -max_eff_mm)
    end_mm   = min(mm_range[1],  max_eff_mm)

    # -- Drawing ticks & labels --
    for mm_tick in range(start_mm, end_mm + 1):
        if mm_tick == 0:
            continue  # center is already drawn

        # Compute pixel offset from center
        if focal_length and iris_depth_mm:
            offset_px = int((mm_tick * focal_length) / (iris_depth_mm * PIXELS_PER_MM))
        else:
            offset_px = int(mm_tick / PIXELS_PER_MM)

        # ----------------------
        # (A) HORIZONTAL TICK
        # ----------------------
        tick_x = cx + offset_px
        if 0 <= tick_x < w:
            # Shorter tick lines, ±4 px
            cv2.line(image, (tick_x, cy - 4), (tick_x, cy + 4), (255, 255, 0), bold_thickness)
            
            # Place the label above or below the axis to reduce overlap
            if mm_tick % 5 == 0:
                if mm_tick > 0:
                    # Positive -> label above the axis
                    label_y = cy - 8
                else:
                    # Negative -> label below the axis
                    label_y = cy + 14

                # Slight horizontal shift so text is roughly centered on the tick
                cv2.putText(
                    image,
                    f"{mm_tick} mm",
                    (tick_x - 14, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # ----------------------
        # (B) VERTICAL TICK
        # ----------------------
        tick_y = cy + offset_px
        if 0 <= tick_y < h:
            cv2.line(image, (cx - 4, tick_y), (cx + 4, tick_y), (255, 255, 0), bold_thickness)
            
            # Typically, if you want "above the center" to read as negative,
            # you might do something like label = f"{-mm_tick} mm"
            # We'll keep that logic below:
            label_val = -mm_tick

            # For vertical labeling, push positives to the right, negatives to the left
            if mm_tick % 5 == 0:
                if mm_tick > 0:
                    label_x = cx + 8
                else:
                    label_x = cx - 40

                cv2.putText(
                    image,
                    f"{label_val} mm",
                    (label_x, tick_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

    return image



def plot_deviation_polar(results, eye="Left", save_path=None):
    """
    Verilen sonuçlardan (results sözlüğü) seçilen göz (Left veya Right)
    için sapma açısını ve büyüklüğünü polar koordinatlarda çizer.
    
    Ölçülen sapma; pupil/iris merkezi ile glare (yansıma) merkezi arasındaki
    farkın yatay ve düşey bileşenlerinin bileşkesiyle hesaplanır.
    Grafik, save_path parametresi verilmemişse temp_reports klasörüne kaydedilir
    ve dosya yolu döndürülür.
    """
    import math
    import matplotlib.pyplot as plt
    pinhole_results = results.get("pinhole_strabismus_results")
    if eye == "Left":
        h_angle = pinhole_results.get("left_h_angle_deg", 0)
        v_angle = pinhole_results.get("left_v_angle_deg", 0)
        deviation_magnitude=pinhole_results.get("left_deviation_magnitude", 0)
    elif eye == "Right":
        h_angle = pinhole_results.get("right_h_angle_deg", 0)
        v_angle = pinhole_results.get("right_v_angle_deg", 0)   
        deviation_magnitude=pinhole_results.get("right_deviation_magnitude", 0)       
        

    else:
        print("Geçersiz göz seçildi! 'Left' veya 'Right' kullanılmalı.")
        return None
    deviation_angle = math.sqrt(h_angle**2 + v_angle**2)
    deviation_angle_rad = math.radians(deviation_angle)
    #gerçek boyutu esas alıyoruz.
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(deviation_angle_rad, deviation_magnitude, 'ro', markersize=10, label="Measured Deviation")
    ax.set_title(f"{eye} Eye Deviation (Polar)")
    ax.legend(loc="lower right")
    
    if save_path is None:
        temp_folder = "temp_reports"
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        save_path = os.path.join(temp_folder, f"deviation_polar_{eye}.png")
    
    fig.savefig(save_path, format="png", bbox_inches="tight")
    plt.close(fig)
    return save_path








def _draw_two_eye_diagrams(
    left_ref=(0,0),
    right_ref=(0,0),
    iris_degree=20  # 20° in each direction from center
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Each axis goes from -iris_degree to +iris_degree
    for ax, eye_name, (rx, ry) in [
        (axes[0], "Right Eye", right_ref),
        (axes[1], "Left Eye",  left_ref),
    ]:
        ax.set_aspect("equal")
        ax.set_xlim(-iris_degree, iris_degree)
        ax.set_ylim(-iris_degree, iris_degree)
        ax.set_title(eye_name, fontsize=12, fontweight='bold')

        # Draw crosshairs for reference
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

        # Optionally, label the extremes:
        ax.text(iris_degree, 0, "Exo →", color="red", ha="left", va="center")
        ax.text(-iris_degree, 0, "← Eso", color="red", ha="right", va="center")
        ax.text(0, iris_degree, "Hyper ↑", color="red", ha="center", va="bottom")
        ax.text(0, -iris_degree, "↓ Hypo", color="red", ha="center", va="top")

        # Plot the star at (rx, ry)
        ax.plot(rx, ry, 'r*', markersize=12, label="Reflected Light")
        ax.text(rx + 0.5, ry + 0.5, f"({rx:.1f}, {ry:.1f})", color="black",
                bbox=dict(facecolor="red", alpha=0.6), fontsize=9)
        ax.legend(loc="lower right")
        ax.grid(True)

    filename = os.path.join("temp_reports", "strabismus_karnesi.png")
    os.makedirs("temp_reports", exist_ok=True)
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)
    fig.savefig(filename)
    plt.close(fig)
    return filename

def plot_strabismus_karnesi_dynamic(left_h_angle, left_v_angle, right_h_angle, right_v_angle, chart_limit=45):
                                    
    filename=_draw_two_eye_diagrams(
        # For the left eye chart, place the star at (left_h_angle, left_v_angle)
        left_ref=(left_h_angle, left_v_angle),
        # For the right eye chart, place the star at (right_h_angle, right_v_angle)
        right_ref=(right_h_angle, right_v_angle),
        iris_degree=chart_limit
    )
    return filename
def overlay_polygon_on_image(image, pts, offset, color, alpha=0.3):
    if image is None:
        return None
    pts_transformed = pts - np.array(offset)  # offset: (x,y)
    overlay = image.copy()
    cv2.fillPoly(overlay, [pts_transformed.astype(np.int32).reshape((-1,1,2))], color)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# FALLBACK Fonksiyonları
def fallback_iris_detection(roi_img):
    print("Fallback Iris Detection triggered")
    try:
        iris_model = IrisLandmark()
        iris_mask, iris_center, iris_radius = iris_model.detect(roi_img)
        
        if iris_mask is not None and iris_center is not None:
            return {
                "mask": iris_mask,
                "center": iris_center,
                "radius": iris_radius
            }
    except Exception as e:
        print("Fallback iris tespitinde hata:", e)
    return None


def fallback_pupil_detection(roi_img, iris_mask=None, iris_center=None, iris_radius=None):
    print("Fallback Pupil Detection triggered")
    try:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray, dtype=np.uint8)
        if iris_mask is not None:
            mask = iris_mask.copy()
        elif iris_center is not None and iris_radius is not None:
            mask = create_mask_for_circle(roi_img.shape, iris_center, iris_radius)
        else:
            mask.fill(255)

        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        _, thresh = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    except Exception as e:
        print("Fallback pupil tespitinde hata:", e)
    return None

def fallback_glare_detection(roi_img, iris_mask=None, iris_center=None, iris_radius=None):
    print("Fallback Glare Detection triggered")
    try:
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # 1) İris bölgesini maskele
        mask = np.zeros_like(hsv[:, :, 0], dtype=np.uint8)
        if iris_mask is not None:
            mask = iris_mask.copy()
        elif iris_center is not None and iris_radius is not None:
            mask = create_mask_for_circle(roi_img.shape, iris_center, iris_radius)
        else:
            mask.fill(255)

        # 2) Parlaklık alanı tespiti (örnek HSV eşikleri)
        lower = np.array([0, 0, 200])
        upper = np.array([180, 50, 255])
        glare_candidates = cv2.inRange(hsv, lower, upper)
        glare_masked = cv2.bitwise_and(glare_candidates, glare_candidates, mask=mask)

        # 3) Morfolojik işlem
        kernel = np.ones((3, 3), np.uint8)
        glare_masked = cv2.morphologyEx(glare_masked, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4) Kontur bul
        contours, _ = cv2.findContours(glare_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)

            # Eğer iris alanı bilgisine sahipseniz, kontrol edin:
            if iris_radius is not None:
                iris_area = cv2.contourArea(iris_mask)
                iris_radius = np.sqrt(iris_area / np.pi)
                if not validate_glare(c, iris_area):
                    print("Tespit edilen glare alanı iris alanına oranla uygun değil.")
                    return None

            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(c)
                cx, cy = x + w // 2, y + h // 2

            pts = c.reshape(-1, 2)
            return {
                "center": (cx, cy),
                "mask": pts
            }
    except Exception as e:
        print("Fallback glare tespitinde hata:", e)
    return None


def plot_detailed_analysis_matplotlib(results):
    """
    Creates a 2x3 grid of panels (for left and right eyes) showing:
      (1) Iris Panel: ROI with Iris overlay.
      (2) Pupil Panel: Extracted iris ROI with Pupil overlay.
      (3) Glare Panel: Extracted iris ROI with Reflection overlay.
    
    The left-eye panels are on the top row and the right-eye panels are on the bottom row.
    This function now also overlays the iris depth values (if available) on the iris panels.
    The generated figure is saved as a PNG file and its path is returned.
    """
    # Retrieve values from the results dictionary.
    left_roi = results.get("left_roi")
    left_eye = results.get("left_eye")            # (x1, y1, x2, y2)
    left_seg = results.get("left_seg_results", {})
    right_roi = results.get("right_roi")
    right_eye = results.get("right_eye")
    right_seg = results.get("right_seg_results", {})

    # --- Left Eye Panels ---
    if left_roi is not None and left_eye is not None:
        left_iris_panel = left_roi.copy()
        if "Iris" in left_seg:
            pts = left_seg["Iris"]["mask"]
            left_iris_panel = overlay_polygon_on_image(left_iris_panel, pts, left_eye[:2], (0,255,255), alpha=0.3)
    else:
        left_iris_panel = np.zeros((300,300,3), dtype=np.uint8)

    # (2) Pupil Panel (Left)
    if left_roi is not None and left_eye is not None and results.get("left_iris_center") is not None and results.get("left_iris_radius", 0) > 0:
        left_center_rel = (results["left_iris_center"][0] - left_eye[0],
                           results["left_iris_center"][1] - left_eye[1])
        left_extracted = IrisAnalyzer.extract_iris_roi(left_roi, left_center_rel, results["left_iris_radius"])
        left_extracted = sharpen_image(left_extracted)
        left_offset = (results["left_iris_center"][0] - results["left_iris_radius"],
                       results["left_iris_center"][1] - results["left_iris_radius"])
    else:
        left_extracted = np.zeros((300,300,3), dtype=np.uint8)
        left_offset = (0,0)
    if "Pupil" in left_seg and "mask" in left_seg["Pupil"]:
        pts = left_seg["Pupil"]["mask"]
        left_pupil_panel = overlay_polygon_on_image(left_extracted, pts, left_offset, (0,255,0), alpha=0.3)

        # Offset-corrected points for drawing
        ellipse_pts = pts - np.array(left_offset)
        if len(ellipse_pts) >= 5:
            ellipse = cv2.fitEllipse(ellipse_pts.astype(np.int32))
            cv2.ellipse(left_pupil_panel, ellipse, (255, 0, 0), 2)  # Blue ellipse
    else:
        left_pupil_panel = left_extracted.copy()


    # (3) Glare Panel (Left)
    if left_roi is not None and left_eye is not None and results.get("left_iris_center") is not None and results.get("left_iris_radius", 0) > 0:
        left_center_rel = (results["left_iris_center"][0] - left_eye[0],
                           results["left_iris_center"][1] - left_eye[1])
        left_extracted_glare = IrisAnalyzer.extract_iris_roi(left_roi, left_center_rel, results["left_iris_radius"])
        left_extracted_glare = sharpen_image(left_extracted_glare)
        left_offset = (results["left_iris_center"][0] - results["left_iris_radius"],
                       results["left_iris_center"][1] - results["left_iris_radius"])
    else:
        left_extracted_glare = np.zeros((300,300,3), dtype=np.uint8)
        left_offset = (0,0)
    if "Reflection" in left_seg:
        pts = left_seg["Reflection"]["mask"]
        left_glare_panel = overlay_polygon_on_image(left_extracted_glare, pts, left_offset, (0,0,255), alpha=0.3)
            # Glare merkezini göster (segment sonucu)
        if "center" in left_seg["Reflection"]:
            cx, cy = left_seg["Reflection"]["center"]
            rel_cx = cx - left_offset[0]
            rel_cy = cy - left_offset[1]
            cv2.drawMarker(left_glare_panel, (int(rel_cx), int(rel_cy)), (0, 255, 255),
                        markerType=cv2.MARKER_STAR, markerSize=4, thickness=1)


        ellipse_pts = pts - np.array(left_offset)
        if len(ellipse_pts) >= 5:
            ellipse = cv2.fitEllipse(ellipse_pts.astype(np.int32))
            cv2.ellipse(left_glare_panel, ellipse, (0, 255, 255), 2)  # Yellow ellipse
            # Optionally overwrite the center for analysis/debug
            left_seg["Reflection"]["center"] = (int(ellipse[0][0] + left_offset[0]), int(ellipse[0][1] + left_offset[1]))
    else:
        left_glare_panel = left_extracted_glare.copy()
    if results.get("left_iris_center") is not None and results.get("left_iris_radius", 0) > 0:
        center_rel = (results["left_iris_center"][0] - left_offset[0],
                      results["left_iris_center"][1] - left_offset[1])
        cv2.circle(left_glare_panel, (int(center_rel[0]), int(center_rel[1])), results["left_iris_radius"], (255,255,255), 2)
    if results.get("left_pupil_center") is not None and left_eye is not None and results.get("left_iris_center") is not None and results.get("left_iris_radius", 0) > 0:
        rel_pupil = (results["left_pupil_center"][0] - left_offset[0],
                     results["left_pupil_center"][1] - left_offset[1])
        cv2.drawMarker(left_pupil_panel, (int(rel_pupil[0]), int(rel_pupil[1])), (0,0,255),
                       markerType=cv2.MARKER_STAR, markerSize=7, thickness=1)

    # --- Right Eye Panels ---
    if right_roi is not None and right_eye is not None:
        right_iris_panel = right_roi.copy()
        if "Iris" in right_seg:
            pts = right_seg["Iris"]["mask"]
            right_iris_panel = overlay_polygon_on_image(right_iris_panel, pts, right_eye[:2], (0,255,255), alpha=0.3)
    else:
        right_iris_panel = np.zeros((300,300,3), dtype=np.uint8)
    if right_roi is not None and right_eye is not None and results.get("right_iris_center") is not None and results.get("right_iris_radius", 0) > 0:
        right_center_rel = (results["right_iris_center"][0] - right_eye[0],
                            results["right_iris_center"][1] - right_eye[1])
        right_extracted = IrisAnalyzer.extract_iris_roi(right_roi, right_center_rel, results["right_iris_radius"])
        right_extracted = sharpen_image(right_extracted)
        right_offset = (results["right_iris_center"][0] - results["right_iris_radius"],
                        results["right_iris_center"][1] - results["right_iris_radius"])
    else:
        right_extracted = np.zeros((300,300,3), dtype=np.uint8)
        right_offset = (0,0)
    if "Pupil" in right_seg and "mask" in right_seg["Pupil"]:
        pts = right_seg["Pupil"]["mask"]
        right_pupil_panel = overlay_polygon_on_image(right_extracted, pts, right_offset, (0,255,0), alpha=0.3)

        # Offset-corrected points for drawing
        ellipse_pts = pts - np.array(right_offset)
        if len(ellipse_pts) >= 5:
            ellipse = cv2.fitEllipse(ellipse_pts.astype(np.int32))
            cv2.ellipse(right_pupil_panel, ellipse, (255, 0, 0), 2)  # Blue ellipse
    else:
        left_pupil_panel = left_extracted.copy()

    if results.get("right_pupil_center") is not None and right_eye is not None and results.get("right_iris_center") is not None and results.get("right_iris_radius", 0) > 0:
        rel_pupil = (results["right_pupil_center"][0] - right_offset[0],
                     results["right_pupil_center"][1] - right_offset[1])
        cv2.drawMarker(right_pupil_panel, (int(rel_pupil[0]), int(rel_pupil[1])), (0,0,255),
                       markerType=cv2.MARKER_STAR, markerSize=7, thickness=1)
    if right_roi is not None and right_eye is not None and results.get("right_iris_center") is not None and results.get("right_iris_radius", 0) > 0:
        right_center_rel = (results["right_iris_center"][0] - right_eye[0],
                            results["right_iris_center"][1] - right_eye[1])
        right_extracted_glare = IrisAnalyzer.extract_iris_roi(right_roi, right_center_rel, results["right_iris_radius"])
        right_extracted_glare = sharpen_image(right_extracted_glare)
        right_offset = (results["right_iris_center"][0] - results["right_iris_radius"],
                        results["right_iris_center"][1] - results["right_iris_radius"])
    else:
        right_extracted_glare = np.zeros((300,300,3), dtype=np.uint8)
        right_offset = (0,0)
    if "Reflection" in right_seg:
        pts = right_seg["Reflection"]["mask"]
        right_glare_panel = overlay_polygon_on_image(right_extracted_glare, pts, right_offset, (0,0,255), alpha=0.3)
        if "center" in right_seg["Reflection"]:
            cx, cy = right_seg["Reflection"]["center"]
            rel_cx = cx - right_offset[0]
            rel_cy = cy - right_offset[1]
            cv2.drawMarker(right_glare_panel, (int(rel_cx), int(rel_cy)), (0, 255, 255),
                        markerType=cv2.MARKER_STAR, markerSize=4, thickness=1)


        ellipse_pts = pts - np.array(right_offset)
        if len(ellipse_pts) >= 5:
            ellipse = cv2.fitEllipse(ellipse_pts.astype(np.int32))
            cv2.ellipse(right_glare_panel, ellipse, (0, 255, 255), 2)  # Yellow ellipse
            right_seg["Reflection"]["center"] = (int(ellipse[0][0] + right_offset[0]), int(ellipse[0][1] + right_offset[1]))
    else:
        right_glare_panel = right_extracted_glare.copy()
    if results.get("right_iris_center") is not None and results.get("right_iris_radius", 0) > 0:
        center_rel = (results["right_iris_center"][0] - right_offset[0],
                      results["right_iris_center"][1] - right_offset[1])
        cv2.circle(right_glare_panel, (int(center_rel[0]), int(center_rel[1])), results["right_iris_radius"], (255,255,255), 2)

    # Resize all panels to a common size.
    common_size = (200,200)
    def resize_panel(img):
        return cv2.resize(img, common_size)
    left_iris_panel = resize_panel(left_iris_panel)
    left_pupil_panel = resize_panel(left_pupil_panel)
    left_glare_panel = resize_panel(left_glare_panel)
    right_iris_panel = resize_panel(right_iris_panel)
    right_pupil_panel = resize_panel(right_pupil_panel)
    right_glare_panel = resize_panel(right_glare_panel)

    # --- NEW: Overlay Iris Depth Text ---
    if results.get("left_iris_depth_cm") is not None:
        depth_text_left = f"Depth: {results['left_iris_depth_cm']:.2f} cm"
        cv2.putText(left_iris_panel, depth_text_left, (5, left_iris_panel.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    if results.get("right_iris_depth_cm") is not None:
        depth_text_right = f"Depth: {results['right_iris_depth_cm']:.2f} cm"
        cv2.putText(right_iris_panel, depth_text_right, (5, right_iris_panel.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)

    # Create composite image from panels.
    top_row = np.hstack([left_iris_panel, left_pupil_panel, left_glare_panel])
    bottom_row = np.hstack([right_iris_panel, right_pupil_panel, right_glare_panel])
    composite = np.vstack([top_row, bottom_row])

    # Plot the panels using matplotlib.
    fig, axes = plt.subplots(2, 3, figsize=(12,8))
    axes[0,0].imshow(cv2.cvtColor(left_iris_panel, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("Right Iris")
    axes[0,0].axis("off")
    axes[0,1].imshow(cv2.cvtColor(left_pupil_panel, cv2.COLOR_BGR2RGB))
    axes[0,1].set_title("Right Pupil")
    axes[0,1].axis("off")
    axes[0,2].imshow(cv2.cvtColor(left_glare_panel, cv2.COLOR_BGR2RGB))
    axes[0,2].set_title("Right Glare\n(Iris Boundary)")
    axes[0,2].axis("off")
    axes[1,0].imshow(cv2.cvtColor(right_iris_panel, cv2.COLOR_BGR2RGB))
    axes[1,0].set_title("Left Iris")
    axes[1,0].axis("off")
    axes[1,1].imshow(cv2.cvtColor(right_pupil_panel, cv2.COLOR_BGR2RGB))
    axes[1,1].set_title("Left Pupil")
    axes[1,1].axis("off")
    axes[1,2].imshow(cv2.cvtColor(right_glare_panel, cv2.COLOR_BGR2RGB))
    axes[1,2].set_title("Left Glare\n(Iris Boundary)")
    axes[1,2].axis("off")
    # Optionally, add legends or additional text on the panels if needed.

    folder = ensure_temp_folder()
    filename = os.path.join(folder, "detailed_analysis.png")

    plt.tight_layout()
    plt.show()
    plt.pause(0.001)
    return save_fig_to_file(fig, filename)

# --- Yardımcı Fonksiyon: Kesikli (Dashed) Çizgi Çizimi ---
def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=10):
    """İki nokta arasında kesikli çizgi çizer."""
    dist = math.hypot(pt2[0]-pt1[0], pt2[1]-pt1[1])
    if dist == 0:
        return
    dash_count = int(dist // dash_length)
    for i in range(dash_count):
        start_fraction = i / dash_count
        end_fraction = (i + 0.5) / dash_count  # her dash segmenti çizgi uzunluğunun yarısı kadar
        start_point = (int(pt1[0] + (pt2[0]-pt1[0]) * start_fraction),
                       int(pt1[1] + (pt2[1]-pt1[1]) * start_fraction))
        end_point = (int(pt1[0] + (pt2[0]-pt1[0]) * end_fraction),
                     int(pt1[1] + (pt2[1]-pt1[1]) * end_fraction))
        cv2.line(img, start_point, end_point, color, thickness)
def plot_roi_with_angles(results):
    import cv2
    import matplotlib.pyplot as plt

    # Pinhole yöntemiyle hesaplanan gerçek şaşılık açısı sonuçlarını alalım:
    pinhole_results = results.get("pinhole_strabismus_results", {})
    left_h_angle = pinhole_results.get("left_h_angle_deg", 0)
    left_v_angle = pinhole_results.get("left_v_angle_deg", 0)
    right_h_angle = pinhole_results.get("right_h_angle_deg", 0)
    right_v_angle = pinhole_results.get("right_v_angle_deg", 0)
    left_angle = math.sqrt(left_h_angle**2 + left_v_angle**2)
    right_angle = math.sqrt(right_h_angle**2 + right_v_angle**2)


    # Debug görüntüsüne de bu bilgiyi ekleyelim (varsa):
    debug_img = results.get("debug_image")
    if debug_img is not None:
        cv2.putText(debug_img, f"Left Strabismus: {left_angle:.2f}°", (30,210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(debug_img, f"Right Strabismus: {right_angle:.2f}°", (30,240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # ROI görüntülerini alalım:
    left_img = results.get("left_roi_with_guidelines")
    right_img = results.get("right_roi_with_guidelines")
    if left_img is None or right_img is None:
        print("ROI görüntülerinden biri alınamadı!")
        return

    # ROI içindeki koordinatlar, orijinal (global) koordinatlar ile elde ediliyorsa,
    # göz bounding box ofsetlerini (left_eye/right_eye) kullanarak ROI içi koordinatlara çeviriyoruz.
    if results.get("left_eye") is not None:
        left_offset = (results["left_eye"][0], results["left_eye"][1])
        if results.get("left_pupil_center") is not None:
            lp = results["left_pupil_center"]
            lp_rel = (lp[0] - left_offset[0], lp[1] - left_offset[1])
            cv2.drawMarker(left_img, (int(lp_rel[0]), int(lp_rel[1])), (0,0,255),
                           markerType=cv2.MARKER_STAR, markerSize=2, thickness=2)
        if results.get("left_ref_center") is not None:
            lr = results["left_ref_center"]
            lr_rel = (lr[0] - left_offset[0], lr[1] - left_offset[1])
            cv2.drawMarker(left_img, (int(lr_rel[0]), int(lr_rel[1])), (255,0,0),
                           markerType=cv2.MARKER_STAR, markerSize=2, thickness=2)

    if results.get("right_eye") is not None:
        right_offset = (results["right_eye"][0], results["right_eye"][1])
        if results.get("right_pupil_center") is not None:
            rp = results["right_pupil_center"]
            rp_rel = (rp[0] - right_offset[0], rp[1] - right_offset[1])
            cv2.drawMarker(right_img, (int(rp_rel[0]), int(rp_rel[1])), (0,0,255),
                           markerType=cv2.MARKER_STAR, markerSize=2, thickness=2)
        if results.get("right_ref_center") is not None:
            rr = results["right_ref_center"]
            rr_rel = (rr[0] - right_offset[0], rr[1] - right_offset[1])
            cv2.drawMarker(right_img, (int(rr_rel[0]), int(rr_rel[1])), (255,0,0),
                           markerType=cv2.MARKER_STAR, markerSize=2, thickness=2)

    # Oluşturulan ROI panellerini matplotlib ile yan yana gösterelim ve pinhole açılarını başlık olarak ekleyelim:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Right ROI\nStrabismus Angle: {left_angle:.2f}°")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Left ROI\nStrabismus Angle: {right_angle:.2f}°")
    axes[1].axis("off")

    folder = ensure_temp_folder()
    filename = os.path.join(folder, "roi_angles.png")
    plt.tight_layout()
    plt.show()
    plt.pause(0.001)
    return save_fig_to_file(fig, filename)

# --- Ek Matplotlib Fonksiyonu: Glare & Pupil Merkez Analizi ---
def plot_centers_analysis_matplotlib(results):
    """
    Bu fonksiyonda:
      - Orijinal (yüksek çözünürlüklü) veya debug görüntü kullanılarak,
      - Sol ve sağ göz için pupil ve glare merkezleri, yıldızla işaretlenir.
      - Aralarına sarı kesikli çizgi çizilir.
      - Her göz için strabismus sapma açısı (horizantal ve vertical açıların bileşkesi) hesaplanır ve görsele eklenir.
    """
    # Mümkünse orijinal görüntü kullanılsın:
    if "orig_image" in results and results["orig_image"] is not None:
        base_img = results["orig_image"].copy()
    else:
        base_img = results["debug_image"].copy()

    pinhole_results = results.get("pinhole_strabismus_results")
    left_img = results.get("left_roi")
    right_img = results.get("right_roi")
    
    if results.get("left_eye") is not None:
        left_offset = (results["left_eye"][0], results["left_eye"][1])
        left_pupil = results["left_pupil_center"]
        left_ref = results["left_ref_center"]
        left_h_angle = pinhole_results.get("left_h_angle_deg", 0)
        left_v_angle = pinhole_results.get("left_v_angle_deg", 0)
        left_deviation = math.sqrt(left_h_angle**2 + left_v_angle**2)
        if left_pupil is not None:
            lp_rel = (left_pupil[0] - left_offset[0], left_pupil[1] - left_offset[1])
            cv2.drawMarker(left_img, (int(lp_rel[0]), int(lp_rel[1])), (0,0,255),
                        markerType=cv2.MARKER_STAR, markerSize=2, thickness=2)
        if left_ref is not None:
            lr_rel = (left_ref[0] - left_offset[0], left_ref[1] - left_offset[1])
            cv2.drawMarker(left_img, (int(lr_rel[0]), int(lr_rel[1])), (255,0,0),
                        markerType=cv2.MARKER_STAR, markerSize=2, thickness=2)
        
        # Use the relative coordinates for drawing the dashed line:
        draw_dashed_line(left_img, (int(lp_rel[0]), int(lp_rel[1])), (int(lr_rel[0]), int(lr_rel[1])), 
                        (0,255,255), thickness=2, dash_length=10)


    if results.get("right_eye") is not None:
        right_offset = (results["right_eye"][0], results["right_eye"][1])
        right_pupil = results["right_pupil_center"]
        right_ref = results["right_ref_center"]
        right_h_angle = pinhole_results.get("right_h_angle_deg", 0)
        right_v_angle = pinhole_results.get("right_v_angle_deg", 0)
        right_deviation = math.sqrt(right_h_angle**2 + right_v_angle**2)
        if right_pupil is not None:
            rp_rel = (right_pupil[0] - right_offset[0], right_pupil[1] - right_offset[1])
            cv2.drawMarker(right_img, (int(rp_rel[0]), int(rp_rel[1])), (0, 0, 255),
                           markerType=cv2.MARKER_STAR, markerSize=2, thickness=2)
        if right_ref is not None:
            rr_rel = (right_ref[0] - right_offset[0], right_ref[1] - right_offset[1])
            cv2.drawMarker(right_img, (int(rr_rel[0]), int(rr_rel[1])), (255, 0, 0),
                           markerType=cv2.MARKER_STAR, markerSize=2, thickness=2)
        # Draw dashed line using relative coordinates:
        draw_dashed_line(right_img, (int(rp_rel[0]), int(rp_rel[1])), (int(rr_rel[0]), int(rr_rel[1])), 
                         (0, 255, 255), thickness=2, dash_length=10)






    #fig, ax = plt.subplots(figsize=(8,6))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Right ROI\nStrabismus Angle: {left_deviation:.2f}°")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Left ROI\nStrabismus Angle: {right_deviation:.2f}°")
    axes[1].axis("off")
    #plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB))
    plt.suptitle("Glare & Pupil Centers with Deviation")
    plt.axis("off")
    folder = ensure_temp_folder()
    filename = os.path.join(folder, "centers_analysis.png")

    plt.show()
    plt.pause(0.001)
    return save_fig_to_file(fig, filename)

def get_focal_length(image_path):
    """
    Extracts the focal length (in mm) from the EXIF data of an image.
    
    Parameters:
        image_path (str): The path to the image file.
    
    Returns:
        float or None: The focal length in millimeters if available, otherwise None.
    """
    try:
        # Open the image file and extract EXIF data
        image = Image.open(image_path)
        exif_data = image._getexif()
    except Exception as e:
        print(f"Error opening image: {e}")
        return None

    if not exif_data:
        print("No EXIF data found.")
        return None

    # Look up the FocalLength tag by name
    focal_length = None
    for tag, value in exif_data.items():
        tag_name = ExifTags.TAGS.get(tag, tag)
        if tag_name == 'FocalLength':
            focal_length = value
            break

    if focal_length is None:
        print("Focal length not found in EXIF data.")
        return None

    print(f"Focal length (EXIF): {focal_length}")
    try:
        # Handle different possible types for the focal length
        if isinstance(focal_length, tuple):
            # Convert tuple (numerator, denominator) to float
            return focal_length[0] / focal_length[1]
        elif isinstance(focal_length, (int, float)):
            return focal_length
        elif isinstance(focal_length, str):
            # Attempt to convert a string to float
            return float(focal_length)
        else:
            # If the object is not one of the expected types, try to cast to float.
            return float(focal_length)
    except Exception as e:
        print(f"Error processing focal length: {e}")
        return None
def load_calibration_data(calibration_file="calibration.pickle"):
    """
    Load the calibration data (including cameraMatrix) from a pickle file.
    """
    try:
        with open(calibration_file, "rb") as f:
            calibration_data = pickle.load(f)
        return calibration_data
    except Exception as e:
        print("Error loading calibration data:", e)
        return None

def compute_pixels_per_mm(f_mm, calibration_data):
    cameraMatrix = calibration_data.get("cameraMatrix")
    if cameraMatrix is None:
        print("Camera matrix not found in calibration data.")
        return None
    fx = cameraMatrix[0, 0]
    print(f"fx: {fx}")
    if f_mm is None:
        print("Focal length not available; using default (35 mm).")
        f_mm = 35.0
    pixels_per_mm = fx / f_mm  # pixels per mm
    return pixels_per_mm

    
def compute_sensor_width_from_calibration(orig_image, f_mm, calibration_data):
    """
    Compute the effective sensor width (in mm) using the camera matrix from calibration.
    
    Parameters:
      - orig_image: the original image as a numpy array.
      - f_mm: the focal length in mm (from EXIF or default).
      - calibration_data: a dict containing calibration results, must include "cameraMatrix".
    
    The camera matrix has fx in pixels. Then:
        pixel_size = f_mm / fx   (mm per pixel)
        sensor_width = image_width * pixel_size
    """
    cameraMatrix = calibration_data.get("cameraMatrix")
    if cameraMatrix is None:
        print("Camera matrix not found in calibration data.")
        return None
    fx = cameraMatrix[0, 0]
    fy= cameraMatrix[1, 1]
    print(f"DEĞER HESAPLAMA fx: {fx}, fy: {fy}")
    image_width = orig_image.shape[1]
    pixel_size = f_mm / fx  # mm per pixel
    sensor_width = image_width * pixel_size
    print(f"Computed sensor width from calibration: {sensor_width:.2f} mm (pixel size: {pixel_size:.4f} mm/pixel)")
    return sensor_width


def estimate_iris_depth(iris_radius, image_width, focal_length, sensor_width, real_iris_diameter=11.7):
    """
    Calculates the iris depth (in cm) using the pinhole camera model.
    
    Parameters:
      - iris_radius (float): Iris radius in pixels.
      - image_width (int): The image width in pixels.
      - focal_length (float): Focal length in millimeters.
      - sensor_width (float): Sensor width in millimeters.
      - real_iris_diameter (float): Real iris diameter in millimeters (default: 11.7 mm).
    
    Calculation:
      1. Compute pixel_size (mm per pixel) = sensor_width / image_width.
      2. Compute the iris diameter in the image (mm) = 2 * iris_radius * pixel_size.
      3. Compute depth (mm) = (focal_length * real_iris_diameter) / (iris_diameter_image_mm).
      4. Convert depth from mm to cm.
    """
    if iris_radius <= 0:
        return None
    pixel_size = sensor_width / image_width  # mm per pixel
    iris_diameter_image_mm = 2 * iris_radius * pixel_size
    depth_mm = (focal_length * real_iris_diameter) / iris_diameter_image_mm
    depth_cm = depth_mm / 10.0
    return depth_cm
def find_glare_center_weighted(roi_img):
    """
    Finds the center of the brightest glare spot in an ROI using intensity-weighted centroid.
    Returns: (center_x, center_y), binary mask of detected glare
    """
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    
    # 1. Threshold to isolate brightest areas (tune threshold if needed)
    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # 2. Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Intensity-weighted centroid
    masked = (gray * (bright_mask // 255)).astype(np.float32)
    M00 = masked.sum()

    if M00 > 0:
        xs = np.arange(masked.shape[1])
        ys = np.arange(masked.shape[0])
        X, Y = np.meshgrid(xs, ys)
        cx = int((X * masked).sum() / M00)
        cy = int((Y * masked).sum() / M00)
        center = (cx, cy)
    else:
        # fallback to contour moments if nothing is bright enough
        cnts, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, bright_mask
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            return None, bright_mask
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center = (cx, cy)

    return center, bright_mask






# ----------------------------------------------------------------------
# ANALİZ FONKSİYONU (YOLO tabanlı)
# ----------------------------------------------------------------------
def analyze_image(image_input, display_in_tk=False, segmentation_method="YOLO"):

    # Orijinal görüntüyü yüksek çözünürlüklü olarak alalım:
    if isinstance(image_input, str):
        orig_image = cv2.imread(image_input)
    else:
        orig_image = image_input.copy()
    if orig_image is None:
        return None
    # İşleme başlamadan önce orijinal görüntüyü saklayalım:
    image = orig_image.copy()
    debug_image = copy.deepcopy(image)
    frontalized = False  # frontalizasyon uygulanıp uygulanmadığını kontrol etmek için

    # Yüz Tespiti ve Head Pose
    face_mesh = FaceMesh(1, 0.7, 0.7)
    headpose_estimator = HeadPoseEstimator()
    iris_analyzer = IrisAnalyzer(None)

    face_results = face_mesh(image)
    if not face_results:
        print("Yüz tespit edilemedi!")
        return None
    
    left_eye = right_eye = None
    left_roi = right_roi = None

    for face_result in face_results:
        head_pose = headpose_estimator.estimate_head_pose(face_result, image)
        if head_pose is not None:
            pitch, yaw, roll = head_pose
        
            norm_roll = normalize_angle(roll)
            cv2.putText(debug_image, f"Old Head Pose: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={norm_roll:.1f}",
                        (30,90), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,255,255), 4)
        else:
            cv2.putText(debug_image, "Old Head Pose: Hesaplanamadı",
                        (30,90), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,255,255), 4)
        left_eye, right_eye = face_mesh.calc_around_eye_bbox(face_result)
        if head_pose is not None and abs(normalize_angle(roll)) > FRONTALIZATION_THRESHOLD:
            image, M = headpose_estimator.frontalize_face(image, left_eye, right_eye)
            debug_image = copy.deepcopy(image)

            cv2.putText(debug_image, f"New Head Pose: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll=0.0",
                        (30,120), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (0,255,255), 4)
            left_eye_pts = np.array([[[left_eye[0], left_eye[1]],
                                       [left_eye[2], left_eye[1]],
                                       [left_eye[2], left_eye[3]],
                                       [left_eye[0], left_eye[3]]]], dtype=np.float32)
            left_eye_trans = cv2.transform(left_eye_pts, M)[0]
            left_eye = (int(np.min(left_eye_trans[:,0])), int(np.min(left_eye_trans[:,1])),
                        int(np.max(left_eye_trans[:,0])), int(np.max(left_eye_trans[:,1])))
            right_eye_pts = np.array([[[right_eye[0], right_eye[1]],
                                        [right_eye[2], right_eye[1]],
                                        [right_eye[2], right_eye[3]],
                                        [right_eye[0], right_eye[3]]]], dtype=np.float32)
            right_eye_trans = cv2.transform(right_eye_pts, M)[0]
            right_eye = (int(np.min(right_eye_trans[:,0])), int(np.min(right_eye_trans[:,1])),
                         int(np.max(right_eye_trans[:,0])), int(np.max(right_eye_trans[:,1])))
            frontalized = True
        # Eğer frontalizasyon yapılmadıysa, orijinal görüntüden ROI’leri alalım
        if not frontalized:
            left_roi = orig_image[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2]].copy()
            right_roi = orig_image[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2]].copy()
        else:
            left_roi = image[left_eye[1]:left_eye[3], left_eye[0]:left_eye[2]].copy()
            right_roi = image[right_eye[1]:right_eye[3], right_eye[0]:right_eye[2]].copy()
        break
    headpose_results ={"pitch":pitch, "yaw":yaw,"roll": roll}
    
    left_offset = (left_eye[0], left_eye[1])
    right_offset = (right_eye[0], right_eye[1])
    
    try:
        yolo_model = YOLO("best_xlarge.pt")
        print("1")
        yolo_model.export(format="onnx", opset=12)
        print("2")
        yolo_model.fuse()
        print("3")
        if torch.cuda.is_available():
            yolo_model.to('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        return debug_image
    
    def segment_roi(roi_img, offset):
        seg_dict = {}
        try:
            results = yolo_model.predict(source=roi_img, conf=0.60)
            if results and len(results) > 0 and results[0].masks is not None:
                try:
                    classes = results[0].masks.cls.cpu().numpy().astype(int)
                except Exception:
                    classes = np.arange(len(results[0].masks.xy))
                for i, mask_pts in enumerate(results[0].masks.xy):
                    pts = mask_pts.astype(np.int32)
                    cls_id = int(classes[i])
                    class_name = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                    color = CLASS_COLORS.get(cls_id, (255,255,255))
                    pts[:,0] += offset[0]
                    pts[:,1] += offset[1]
                    cv2.polylines(debug_image, [pts.reshape((-1,1,2))], True, color, 2)
                    #cv2.putText(debug_image, f"YOLO {class_name}",
                    #            (pts[0][0], pts[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    if class_name == "Reflection":
                        # Extract local ROI of the glare region
                        glare_mask_local = np.zeros_like(roi_img[:, :, 0])
                        roi_polygon = pts.copy() - np.array(offset)
                        cv2.fillPoly(glare_mask_local, [roi_polygon], 255)

                        glare_roi = cv2.bitwise_and(roi_img, roi_img, mask=glare_mask_local)
                        weighted_center, _ = find_glare_center_weighted(glare_roi)

                        if weighted_center:
                            cx, cy = weighted_center[0] + offset[0], weighted_center[1] + offset[1]
                        else:
                            if len(pts) >= 5:
                                ellipse = cv2.fitEllipse(pts)
                                cx, cy = map(int, ellipse[0])
                            else:
                                cx, cy = compute_polygon_center(pts)
                    else:
                        if len(pts) >= 5:
                            ellipse = cv2.fitEllipse(pts)
                            cx, cy = map(int, ellipse[0])
                        else:
                            cx, cy = compute_polygon_center(pts)


                    cv2.circle(debug_image, (cx, cy), 4, color, -1)
                    seg_dict[class_name] = {"mask": pts, "center": (cx, cy)}
        except Exception as e:
            print("Segmentasyon hatası (ROI):", e)
        return seg_dict

    left_offset = (left_eye[0], left_eye[1])
    right_offset = (right_eye[0], right_eye[1])
    left_seg_results = segment_roi(left_roi, left_offset)
    right_seg_results = segment_roi(right_roi, right_offset)
    
    left_iris_center = None
    left_iris_radius = 0
    left_ref_center = None
    left_pupil_center = None
    if "Iris" in left_seg_results:
        left_iris_seg_method = "Segmentasyon"
        print("Sol gözde iris algılandı")
        iris_pts = left_seg_results["Iris"]["mask"]
        (ix, iy), iris_r = cv2.minEnclosingCircle(iris_pts.reshape(-1,2))
        left_iris_center = (int(ix), int(iy))
        left_iris_radius = int(iris_r)
    if "Iris" not in left_seg_results:
        left_iris_seg_method = "Fallback"
        print("Sol gözde iris algılanamadı")
        fallback = fallback_iris_detection(left_roi)
        if fallback is not None:
            left_seg_results["Iris"] = fallback


    # =========================
    #   SOL GÖZ => GLARE
    # =========================
    if "Reflection" in left_seg_results:
        left_ref_seg_method = "Segmentasyon"
        left_ref_center = left_seg_results["Reflection"]["center"]
    else:
        print("Sol gözde yansıma algılanamadı, fallback başlatılıyor")
        # fallback glare => iris bölgesi içinde
        iris_mask_pts = None
        fallback_iris_c = None
        fallback_iris_r = None

        if "Iris" in left_seg_results:
            if isinstance(left_seg_results["Iris"].get("mask"), np.ndarray):
                iris_mask_pts = left_seg_results["Iris"]["mask"]
            fallback_iris_c = left_seg_results["Iris"].get("center")
            fallback_iris_r = left_seg_results["Iris"].get("radius")

        local_roi = left_roi.copy()
        if iris_mask_pts is not None:
            local_polygon = iris_mask_pts.copy()
            local_polygon[:,0] -= left_eye[0]
            local_polygon[:,1] -= left_eye[1]
            glare_center_local = fallback_glare_detection(
                local_roi,
                iris_mask=create_mask_for_polygon(local_roi.shape, local_polygon)
            )
        else:
            if fallback_iris_c is not None and fallback_iris_r is not None:
                local_c = (fallback_iris_c[0] - left_eye[0], fallback_iris_c[1] - left_eye[1])
                glare_center_local = fallback_glare_detection(
                    local_roi,
                    iris_center=local_c,
                    iris_radius=fallback_iris_r
                )
            else:
                glare_center_local = fallback_glare_detection(local_roi)

        if glare_center_local is not None:
            # Bu artık dict döndürüyor: {"center": (cx,cy), "mask": pts}
            left_ref_seg_method = "Fallback"
            local_center = glare_center_local["center"]      # ROI içi merkez
            local_polygon = glare_center_local["mask"]       # ROI içi poligon

            # Global koordinata çevirmek için offset ekle
            global_center = (local_center[0] + left_eye[0],
                            local_center[1] + left_eye[1])
            global_polygon = local_polygon.copy()
            global_polygon[:,0] += left_eye[0]
            global_polygon[:,1] += left_eye[1]

            left_ref_center = global_center
            left_seg_results["Reflection"] = {
                "center": left_ref_center,
                "mask": global_polygon
            }
        else:
            left_ref_seg_method = "Başarısız"


    

    if "Pupil" in left_seg_results:
        print("Sol gözde pupil algılandı")
        left_pupil_seg_method = "Segmentasyon"
        left_pupil_center = left_seg_results["Pupil"]["center"]

    if "Pupil" not in left_seg_results:
        print("Sol gözde pupil algılanamadı")
        left_pupil_center = fallback_pupil_detection(left_roi)
        if left_pupil_center is not None:
            left_seg_results["Pupil"] = {"center": left_pupil_center}
            left_pupil_seg_method = "Fallback"
        else:
            left_pupil_seg_method = "Başarısız"


    right_iris_center = None
    right_iris_radius = 0
    right_ref_center = None
    right_pupil_center = None
    if "Iris" in right_seg_results:
        right_iris_seg_method = "Segmentasyon"
        iris_pts = right_seg_results["Iris"]["mask"]
        (ix, iy), iris_r = cv2.minEnclosingCircle(iris_pts.reshape(-1, 2))
        right_iris_center = (int(ix), int(iy))
        right_iris_radius = int(iris_r)
    else:
        print("Sağ gözde iris algılanamadı")
        fallback = fallback_iris_detection(right_roi)
        if fallback is not None:
            right_seg_results["Iris"] = fallback
            right_iris_seg_method = "Fallback"
            right_iris_center = fallback.get("center")
            right_iris_radius = fallback.get("radius", 0)
        else:
            right_iris_seg_method = "Başarısız"
            right_iris_center = None
            right_iris_radius = 0





    
    if "Reflection" in right_seg_results:
        right_ref_seg_method = "Segmentasyon"
        right_ref_center = right_seg_results["Reflection"]["center"]
        print("Sağ gözde yansıma algılandı")
    else:
        # fallback glare => iris bölgesi içinde
        print("Sağ gözde yansıma algılanamadı, fallback başlatılıyor")
        iris_mask_pts = None
        fallback_iris_c = None
        fallback_iris_r = None

        if "Iris" in right_seg_results:
            if isinstance(right_seg_results["Iris"].get("mask"), np.ndarray):
                iris_mask_pts = right_seg_results["Iris"]["mask"]
            fallback_iris_c = right_seg_results["Iris"].get("center")
            fallback_iris_r = right_seg_results["Iris"].get("radius")
        else:
            print("Sağ gözde iris algılanamadı (yansıma için fallback yapılıyor)")
        local_roi = right_roi.copy()
        if iris_mask_pts is not None:
            local_polygon = iris_mask_pts.copy()
            local_polygon[:,0] -= right_eye[0]
            local_polygon[:,1] -= right_eye[1]
            glare_center_local = fallback_glare_detection(
                local_roi,
                iris_mask=create_mask_for_polygon(local_roi.shape, local_polygon)
            )
        else:
            if fallback_iris_c is not None and fallback_iris_r is not None:
                local_c = (fallback_iris_c[0] - right_eye[0], fallback_iris_c[1] - right_eye[1])
                glare_center_local = fallback_glare_detection(
                    local_roi,
                    iris_center=local_c,
                    iris_radius=fallback_iris_r
                )
            else:
                glare_center_local = fallback_glare_detection(local_roi)

        if glare_center_local is not None:
            # Bu artık dict döndürüyor: {"center": (cx,cy), "mask": pts}
            right_ref_seg_method = "Fallback"
            local_center = glare_center_local["center"]      # ROI içi merkez
            local_polygon = glare_center_local["mask"]       # ROI içi poligon

            # Global koordinata çevirmek için offset ekle
            global_center = (local_center[0] + right_eye[0],
                            local_center[1] + right_eye[1])
            global_polygon = local_polygon.copy()
            global_polygon[:,0] += right_eye[0]
            global_polygon[:,1] += right_eye[1]

            right_ref_center = global_center
            right_seg_results["Reflection"] = {
                "center": right_ref_center,
                "mask": global_polygon
            }
            print("Fallback yansıma tespiti başarılı")
        else:
            right_ref_seg_method = "Başarısız"
            print("Fallback yansıma tespiti başarısız")

    if "Pupil" in right_seg_results:
        print("Sağ gözde pupil algılandı")
        right_pupil_seg_method = "Segmentasyon"
        right_pupil_center = right_seg_results["Pupil"]["center"]
    else:
        print("Sağ gözde pupil algılanamadı, fallback başlatılıyor")
        right_pupil_center = fallback_pupil_detection(right_roi)  # <-- DÜZELTİLMİŞ
        if right_pupil_center is not None:
            right_pupil_seg_method = "Fallback"
            right_seg_results["Pupil"] = {"center": right_pupil_center}
            print("Fallback pupil tespiti başarılı")
        else:
            right_pupil_seg_method = "Başarısız"
            print("Fallback pupil tespiti başarısız")


    left_h_angle = left_v_angle = 0.0
    right_h_angle = right_v_angle = 0.0
    if left_iris_center and left_iris_radius > 0 and left_ref_center:
        dx = left_ref_center[0] - left_iris_center[0]
        dy = left_ref_center[1] - left_iris_center[1]
        left_width = 2 * left_iris_radius
        left_h_angle = (dx / left_width) * MAX_ANGLE_PER_RADIUS * (-1)
        left_v_angle = (dy / left_width) * MAX_ANGLE_PER_RADIUS * (-1)
    if right_iris_center and right_iris_radius > 0 and right_ref_center:
        dx = right_ref_center[0] - right_iris_center[0]
        dy = right_ref_center[1] - right_iris_center[1]
        right_width = 2 * right_iris_radius
        right_h_angle = (dx / right_width) * MAX_ANGLE_PER_RADIUS * (-1)
        right_v_angle = (dy / right_width) * MAX_ANGLE_PER_RADIUS * (-1)

    left_res_text = iris_analyzer.classify_strabismus(left_h_angle, left_v_angle, eye_side="Left")
    right_res_text = iris_analyzer.classify_strabismus(right_h_angle, right_v_angle, eye_side="Right")

    if left_iris_center:
        cv2.circle(debug_image, left_iris_center, 3, (255,0,0), -1)
        if left_ref_center:
            cv2.circle(debug_image, left_ref_center, 5, (0,0,255), -1)
            cv2.line(debug_image, left_iris_center, left_ref_center, (0,255,255), 2)
    if left_pupil_center:
        cv2.circle(debug_image, left_pupil_center, 5, (255,255,0), -1)
    if right_iris_center:
        cv2.circle(debug_image, right_iris_center, 3, (255,0,0), -1)
        if right_ref_center:
            cv2.circle(debug_image, right_ref_center, 5, (0,0,255), -1)
            cv2.line(debug_image, right_iris_center, right_ref_center, (0,255,255), 2)
    if right_pupil_center:
        cv2.circle(debug_image, right_pupil_center, 5, (255,255,0), -1)

    cv2.putText(debug_image, left_res_text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 2)
    cv2.putText(debug_image, right_res_text, (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255,255,255), 2)
    
    strabismus_chart_image_path = None  # Initialize the variable

    

    if left_iris_center is not None and left_eye is not None:
        left_iris_center_local = (left_iris_center[0] - left_eye[0],
                                left_iris_center[1] - left_eye[1])
    else:
        left_iris_center_local = None
        print("Sol iris merkezi bulunamadı.")

    if right_iris_center is not None and right_eye is not None:
        right_iris_center_local = (right_iris_center[0] - right_eye[0],
                                right_iris_center[1] - right_eye[1])
    else:
        right_iris_center_local = None
        print("Sağ iris merkezi bulunamadı.")

    focal = get_focal_length(image_input)

    # Load calibration data
    calibration_file = os.path.join("kalibrasyon", "calibration.pickle")
    calibration_data = load_calibration_data(calibration_file)

    

    iris_detector = IrisLandmark()
    iris_analyzer = IrisAnalyzer(iris_detector)
    left_iris, right_iris = iris_analyzer.detect_iris(image, left_eye, right_eye)
    left_center, left_radius = iris_analyzer.calc_iris_center_and_radius(left_iris)
    right_center, right_radius = iris_analyzer.calc_iris_center_and_radius(right_iris)


    # Assume orig_image is loaded already
    image_height, image_width = orig_image.shape[:2]
    image_width_pixels = image_width
    # Get focal length (in mm) from EXIF
    focal_length_mm = get_focal_length(image_input)
    if focal is None:
        print("Focal length not available; using default (35 mm).")
        focal = 35.0

    

    if calibration_data is None:
        sensor_width = 5.5  # mm, varsayılan değer
    else:
        sensor_width = compute_sensor_width_from_calibration(orig_image, focal_length_mm, calibration_data)
        if sensor_width is None:
            sensor_width = 5.5


    import math

    def analyze_strabismus_with_pinhole(
        left_iris_radius,
        right_iris_radius,
        sensor_width_mm,
        image_width_pixels,
        focal_length_mm,
        left_iris_depth_mm,
        right_iris_depth_mm,
        left_pupil_center,
        right_pupil_center,
        left_ref_center,
        right_ref_center,
        threshold=0.5
    ):
        """
        Pinhole kamera modeline göre, gerçek interpupiller mesafe (IPD) ve
        her gözdeki glare (yansıma) ile pupil merkezi arasındaki gerçek sapmayı hesaplar.
        Sonrasında, bu değerlere dayanarak şaşılık açısını (horizontal/vertical) elde eder
        ve classify_strabismus fonksiyonuna benzer bir şekilde Exotropia/Esotropia vs.
        tanılamasını yapar.

        returns:
        a dict containing:
            - real_ipd_mm
            - left_h_angle_deg, left_v_angle_deg
            - right_h_angle_deg, right_v_angle_deg
            - left_classification (str)
            - right_classification (str)
        """


        # 1) Ölçülen pupil mesafesini (piksel cinsinden) hesaplayalım
        dx = right_pupil_center[0] - left_pupil_center[0]
        dy = right_pupil_center[1] - left_pupil_center[1]
        pupil_distance_pixels = math.sqrt(dx**2 + dy**2)

        # 2) Piksel başına fiziksel boyut (mm/piksel)
        pixel_size = sensor_width_mm / image_width_pixels

        # 3) Pinhole modeli ile Gerçek IPD
        measured_ipd_mm = pupil_distance_pixels * pixel_size
        real_ipd_mm = (measured_ipd_mm * left_iris_depth_mm) / focal_length_mm
        
        

        # Bu fonksiyon X/Y piksel farklarını "gerçek (mm)" cinsine çevirir
        def compute_xy_deviation_mm(glare_center, pupil_center,iris_depth_mm):
            dx_pixels = glare_center[0] - pupil_center[0]
            dy_pixels = glare_center[1] - pupil_center[1]
            dx_mm = dx_pixels * pixel_size
            dy_mm = dy_pixels * pixel_size
            # Pinhole ile gerçeğe dönüştür:
            real_dx = (dx_mm * iris_depth_mm) / focal_length_mm
            real_dy = (dy_mm * iris_depth_mm) / focal_length_mm
            return real_dx, real_dy
        left_iris_radius_mm = left_iris_radius * pixel_size * (left_iris_depth_mm / focal_length_mm)
        right_iris_radius_mm = right_iris_radius * pixel_size* (right_iris_depth_mm / focal_length_mm)
        print(f"Sol iris yarıçapı (mm): {left_iris_radius_mm:.2f}")
        print(f"Sağ iris yarıçapı (mm): {right_iris_radius_mm:.2f}")
        average_depth = (left_iris_depth_mm+right_iris_depth_mm)/2
        print(f"Ortalama iris derinliği (mm): {average_depth:.2f}")

        # 4) Sol göz için X/Y sapmalar (mm)
        left_dx_mm, left_dy_mm = compute_xy_deviation_mm(left_ref_center, left_pupil_center,average_depth)
        # Sağ göz için X/Y sapmalar (mm)
        right_dx_mm, right_dy_mm = compute_xy_deviation_mm(right_ref_center, right_pupil_center,average_depth)

        left_deviation_magnitude= math.sqrt(left_dx_mm**2 + left_dy_mm**2)
        right_deviation_magnitude = math.sqrt(right_dx_mm**2 + right_dy_mm**2)
        print(f"Sol göz sapma büyüklüğü (mm): {left_deviation_magnitude:.2f}")
        print(f"Sağ göz sapma büyüklüğü (mm): {right_deviation_magnitude:.2f}")

        # 5) Her yönde açıyı hesaplayalım (arctan(dev / IPD))
        #    Not: arctan küçük açılar için dev/IPD ≈ açı(radyan)
        
        #linear calculation:
        left_PD=left_deviation_magnitude*15
        right_PD = right_deviation_magnitude*15
        left_h_angle_deg =left_dx_mm*8.55
        left_v_angle_deg = left_dy_mm*8.55
        right_h_angle_deg = right_dx_mm*8.55
        right_v_angle_deg = right_dy_mm*8.55
        print(f"Sol göz yatay açı: {left_h_angle_deg:.2f}°")
        print(f"Sol göz dikey açı: {left_v_angle_deg:.2f}°")
        print(f"Sağ göz yatay açı: {right_h_angle_deg:.2f}°")
        print(f"Sağ göz dikey açı: {right_v_angle_deg:.2f}°")
        """
        left_h_angle_deg = (left_dx_mm / left_iris_radius_mm) * 45.0  # Horizontal deviation for left eye
        left_v_angle_deg = (left_dy_mm / left_iris_radius_mm) * 45.0  # Vertical deviation for left eye

        
        right_h_angle_deg = (right_dx_mm / right_iris_radius_mm) * 45.0  # Horizontal deviation for right eye
        right_v_angle_deg = (right_dy_mm / right_iris_radius_mm) * 45.0  # Vertical deviation for right eye
        
        print(f"Sol göz yatay açı: {left_h_angle_deg:.2f}°")
        print(f"Sol göz dikey açı: {left_v_angle_deg:.2f}°")
        print(f"Sağ göz yatay açı: {right_h_angle_deg:.2f}°")
        print(f"Sağ göz dikey açı: {right_v_angle_deg:.2f}°")
        """
        """
        #angular calculation:
        left_h_angle_deg = math.degrees(math.atan2(left_dx_mm, real_ipd_mm))
        left_v_angle_deg = math.degrees(math.atan2(left_dy_mm, real_ipd_mm))
        right_h_angle_deg = math.degrees(math.atan2(right_dx_mm, real_ipd_mm))
        right_v_angle_deg = math.degrees(math.atan2(right_dy_mm, real_ipd_mm))
        print(f"Sol göz yatay açı: {left_h_angle_deg:.2f}°")
        print(f"Sol göz dikey açı: {left_v_angle_deg:.2f}°")
        print(f"Sağ göz yatay açı: {right_h_angle_deg:.2f}°")
        print(f"Sağ göz dikey açı: {right_v_angle_deg:.2f}°")
        """
        



        def classify_strabismus(h_angle, v_angle, threshold=0.5, eye_side=""):
            results = []
            if abs(h_angle) > threshold:
                if h_angle > 0:
                    results.append(f"Exotropia ~ {abs(h_angle):.1f}°")
                else:
                    results.append(f"Esotropia ~ {abs(h_angle):.1f}°")
            if abs(v_angle) > threshold:
                if v_angle > 0:
                    results.append(f"Hypertropia ~ {abs(v_angle):.1f}°")
                else:
                    results.append(f"Hypotropia ~ {abs(v_angle):.1f}°")
            if not results:
                return "Ortho"
            return "\n".join(results)


        left_classification  = classify_strabismus(left_h_angle_deg,  left_v_angle_deg,  threshold, eye_side="Left")
        right_classification = classify_strabismus(right_h_angle_deg, right_v_angle_deg, threshold, eye_side="Right")

        # 7) Sonuçları bir dict olarak döndürelim
        return {
            "real_ipd_mm": real_ipd_mm,
            "left_h_angle_deg": left_h_angle_deg,
            "left_v_angle_deg": left_v_angle_deg,
            "right_h_angle_deg": right_h_angle_deg,
            "right_v_angle_deg": right_v_angle_deg,
            "left_classification": left_classification,
            "right_classification": right_classification,
            "left_deviation_magnitude": left_deviation_magnitude,
            "right_deviation_magnitude": right_deviation_magnitude,
            "left_PD":left_PD,
            "right_PD":right_PD

        }


    

        
        


    # Open the image file and extract EXIF data
    try:
        exif_Image = Image.open(image_input)
        exif_data = exif_Image._getexif()
        camera_brand = exif_data.get(271)
        camera_model = exif_data.get(272)
        exposure_time = exif_data.get(33434)
        aperture = exif_data.get(33437)
        iso_value = exif_data.get(34855)
        print(f"Kamera Markası: {camera_brand}, Model: {camera_model}")
        print(f"Enstantane: {exposure_time}, Diyafram: {aperture}, ISO: {iso_value}")
    except Exception as e:
        print(f"EXIF verileri alınamadı: {e}")

    # Update results with iris depth using the new functions:
    
    left_iris_depth_cm = estimate_iris_depth(left_iris_radius,image_width, focal, sensor_width)
    print(f"left iris cm: {left_iris_depth_cm:.2f} cm")
    right_iris_depth_cm = estimate_iris_depth(right_iris_radius,image_width, focal, sensor_width)
    print(f"right iris cm: {right_iris_depth_cm:.2f} cm")
    real_distance = compute_interPupilDistance_pinholes(left_pupil_center, right_pupil_center, sensor_width, image_width_pixels, focal_length_mm,left_iris_depth_cm*10)
    print(f"Computed Interpupillary Distance: {real_distance:.2f} mm")

    pinhole_strabismus_results = analyze_strabismus_with_pinhole(
        left_iris_radius,
        right_iris_radius,
        sensor_width,                # sensor_width_mm (örneğin, 5.5 veya hesaplanmış değer)
        image_width_pixels,
        focal_length_mm,
        left_iris_depth_cm * 10,  # iris_depth_mm (cm -> mm dönüşümü)
        right_iris_depth_cm * 10,
        left_pupil_center,
        right_pupil_center,
        left_ref_center,
        right_ref_center
    )
    # (İsteğe bağlı) Karnesi çizimi:
    if left_iris_center and right_iris_center and left_iris_radius > 0 and right_iris_radius > 0  and left_ref_center and right_ref_center:
        strabismus_chart_image_path = plot_strabismus_karnesi_dynamic(
                pinhole_strabismus_results.get("left_h_angle_deg"), pinhole_strabismus_results.get("left_v_angle_deg"),
                pinhole_strabismus_results.get("right_h_angle_deg"), pinhole_strabismus_results.get("right_v_angle_deg"),
    chart_limit=20
        )
    average_depth = (left_iris_depth_cm+right_iris_depth_cm)/2
    
    # Diğer çizim işlemleriniz bittikten sonra

    left_roi_with_guidelines = draw_clinical_guidelines(orig_image,left_roi.copy(), ref_center=left_iris_center_local, focal_length=focal, calibration_data=calibration_data,iris_depth_mm=left_iris_depth_cm)
    right_roi_with_guidelines = draw_clinical_guidelines(orig_image,right_roi.copy(), ref_center=right_iris_center_local, focal_length=focal, calibration_data=calibration_data,iris_depth_mm=right_iris_depth_cm)



    
    print("SOL GOZ CAP   :" +str(left_radius))


    print("SAG GOZ CAP   :" +str(right_radius))


    results_dict = {
        "debug_image": debug_image,
        "orig_image": orig_image,
        "left_roi": left_roi,
        "right_roi": right_roi,
        "left_seg_results": left_seg_results,
        "right_seg_results": right_seg_results,
        "left_iris_center": left_iris_center,
        "left_iris_radius": left_iris_radius,
        "left_ref_center": left_ref_center,
        "left_pupil_center": left_pupil_center,
        "left_iris_depth_cm": left_iris_depth_cm,
        "right_iris_depth_cm": right_iris_depth_cm,
        "right_iris_center": right_iris_center,
        "right_iris_radius": right_iris_radius,
        "right_ref_center": right_ref_center,
        "right_pupil_center": right_pupil_center,
        "PUPIL_DISTANCE":real_distance,
        "left_h_angle": left_h_angle,
        "left_v_angle": left_v_angle,
        "right_h_angle": right_h_angle,
        "right_v_angle": right_v_angle,
        "left_eye": left_eye,
        "right_eye": right_eye,
        "head_pose": head_pose,
        "frontalized": frontalized,
        "left_roi_with_guidelines": left_roi_with_guidelines,
        "right_roi_with_guidelines": right_roi_with_guidelines,
        "segmentation_method": segmentation_method,
        "iris_segmentation": "Sol Göz: " + left_iris_seg_method + ", Sağ Göz: " + right_iris_seg_method,
        "pupil_segmentation": "Sol Göz: " + left_pupil_seg_method + ", Sağ Göz: " + right_pupil_seg_method,
        "reflection_segmentation": "Sol Göz: " + left_ref_seg_method + ", Sağ Göz: " + right_ref_seg_method,
        "strabismus_chart_image":strabismus_chart_image_path,
        "pinhole_strabismus_results": pinhole_strabismus_results,
        "focal_length": focal_length_mm,
        "sensor_width": sensor_width,
        "camera_brand": camera_brand,
        "camera_model": camera_model,
        "exposure_time": exposure_time,
        "aperture": aperture,
        "iso_value": iso_value,
        "headpose_results": headpose_results,
        "iris_depth": average_depth,
    }
    screen_res = (768, 1018)  # Adjust this to your screen resolution
    scale_width = screen_res[0] / debug_image.shape[1]
    scale_height = screen_res[1] / debug_image.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(debug_image.shape[1] * scale)
    window_height = int(debug_image.shape[0] * scale)
    resized_image = cv2.resize(debug_image, (window_width, window_height))

    


