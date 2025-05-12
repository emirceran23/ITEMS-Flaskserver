#!/usr/bin/env python
"""
iris.py
--------
Bu modül, iris tespiti, pupil segmentasyonu ve strabismus analizi işlevlerini gerçekleştiren
IrisAnalyzer sınıfını ve MAX_ANGLE_PER_RADIUS sabitini içerir.
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge

# Sabit parametre: Yaklaşık iris çapı üzerinden Hirschberg açısı hesaplamasında kullanılan katsayı
MAX_ANGLE_PER_RADIUS = 22.0

class IrisAnalyzer:
    def __init__(self, iris_detector):
        self.iris_detector = iris_detector

    def detect_iris(self, image, left_eye_bbox, right_eye_bbox):
        image_height, image_width = image.shape[:2]
        input_shape = self.iris_detector.get_input_shape()
        lx1 = max(left_eye_bbox[0], 0)
        ly1 = max(left_eye_bbox[1], 0)
        lx2 = min(left_eye_bbox[2], image_width)
        ly2 = min(left_eye_bbox[3], image_height)
        left_eye_image = image[ly1:ly2, lx1:lx2].copy()
        eye_contour_left, iris_left = self.iris_detector(left_eye_image)
        print("eye_contour_left:", eye_contour_left)
        left_iris = self.calc_iris_point(left_eye_bbox, iris_left, input_shape)
        rx1 = max(right_eye_bbox[0], 0)
        ry1 = max(right_eye_bbox[1], 0)
        rx2 = min(right_eye_bbox[2], image_width)
        ry2 = min(right_eye_bbox[3], image_height)
        right_eye_image = image[ry1:ry2, rx1:rx2].copy()
        eye_contour_right, iris_right = self.iris_detector(right_eye_image)
        right_iris = self.calc_iris_point(right_eye_bbox, iris_right, input_shape)
        return left_iris, right_iris

    @staticmethod
    def calc_iris_point(eye_bbox, iris, input_shape):
        iris_list = []
        for index in range(5):
            point_x = int(iris[index*3] * ((eye_bbox[2]-eye_bbox[0]) / input_shape[0]))
            point_y = int(iris[index*3+1] * ((eye_bbox[3]-eye_bbox[1]) / input_shape[1]))
            point_x += eye_bbox[0]
            point_y += eye_bbox[1]
            iris_list.append((point_x, point_y))
        return iris_list

    @staticmethod
    def calc_iris_center_and_radius(landmark_list):
        pts = np.array(landmark_list, dtype=np.float32)
        center = np.mean(pts, axis=0).astype(int)
        distances = np.linalg.norm(pts - center, axis=1)
        radius = int(np.max(distances))
        return (int(center[0]), int(center[1])), radius

    @staticmethod
    def draw_debug_image(debug_image, left_iris, right_iris, left_center, left_radius, right_center, right_radius):
        cv2.circle(debug_image, left_center, left_radius, (0,255,0), 2)
        cv2.circle(debug_image, right_center, right_radius, (0,255,0), 2)
        for point in left_iris:
            cv2.circle(debug_image, point, 2, (0,0,255), -1)
        for point in right_iris:
            cv2.circle(debug_image, point, 2, (0,0,255), -1)
        cv2.putText(debug_image, 'r:'+str(left_radius)+'px',
                    (left_center[0]+int(left_radius*1.5), left_center[1]+int(left_radius*0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
        cv2.putText(debug_image, 'r:'+str(right_radius)+'px',
                    (right_center[0]+int(right_radius*1.5), right_center[1]+int(right_radius*0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
        return debug_image

    @staticmethod
    def get_segmentation_images(eye_gray):
        _, bin_img = cv2.threshold(eye_gray, 230, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        morph_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
        morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        return bin_img, morph_img

    @staticmethod
    def detect_corneal_reflection_robust(eye_gray, iris_center=None, max_dist=50, debug_prefix=None):
        _, bin_img = cv2.threshold(eye_gray, 230, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        morph_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)
        morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(morph_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        if len(contours)==1 or iris_center is None:
            c_max = max(contours, key=cv2.contourArea)
            (cx,cy), _ = cv2.minEnclosingCircle(c_max)
            if debug_prefix:
                print(f"{debug_prefix}: Tek kontur bulundu, cx={cx}, cy={cy}")
            return (int(cx), int(cy))
        ix, iy = iris_center
        min_dist = float('inf')
        chosen_center = None
        for c in contours:
            (cx,cy), _ = cv2.minEnclosingCircle(c)
            dist = math.hypot(cx-ix, cy-iy)
            if dist < min_dist and dist < max_dist:
                min_dist = dist
                chosen_center = (int(cx), int(cy))
        if debug_prefix:
            print(f"{debug_prefix}: Seçilen yansıma noktası = {chosen_center}")
        return chosen_center

    @staticmethod
    def classify_strabismus(h_angle, v_angle, threshold=0.5, eye_side=""):
        results = []
        if abs(h_angle) > threshold:
            if h_angle > 0:
                results.append(f"Exotropia ~ {abs(h_angle):.1f} deg")
            else:
                results.append(f"Esotropia ~ {abs(h_angle):.1f} deg")
        if abs(v_angle) > threshold:
            if v_angle > 0:
                results.append(f"Hypertropia ~ {abs(v_angle):.1f} deg")
            else:
                results.append(f"Hypotropia ~ {abs(v_angle):.1f} deg")
        if not results:
            return f"{eye_side}: Ortho"
        return f"{eye_side}: " + " | ".join(results)

    @staticmethod
    def extract_iris_roi(eye_roi, center_local, radius):
        x, y = center_local
        r = radius
        h, w = eye_roi.shape[:2]
        x1 = max(x - r, 0)
        y1 = max(y - r, 0)
        x2 = min(x + r, w)
        y2 = min(y + r, h)
        return eye_roi[y1:y2, x1:x2]

    @staticmethod
    def get_pupil_segmentation_image(iris_roi):
        if iris_roi is None or iris_roi.size == 0:
            return None
        iris_gray = cv2.cvtColor(iris_roi, cv2.COLOR_BGR2GRAY)
        _, pupil_mask = cv2.threshold(iris_gray, 50, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3,3), np.uint8)
        pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return pupil_mask

    @staticmethod
    def overlay_pupil_segmentation(iris_roi, pupil_mask):
        overlay = iris_roi.copy()
        colored_mask = np.zeros_like(iris_roi, dtype=np.uint8)
        colored_mask[pupil_mask>0] = (0,0,255)
        result = cv2.addWeighted(iris_roi, 0.7, colored_mask, 0.3, 0)
        return result

    @staticmethod
    def plot_pupil_segmentation(left_iris_roi, right_iris_roi, left_pupil_mask, right_pupil_mask):
        left_overlay = IrisAnalyzer.overlay_pupil_segmentation(left_iris_roi, left_pupil_mask)
        right_overlay = IrisAnalyzer.overlay_pupil_segmentation(right_iris_roi, right_pupil_mask)
        fig, axs = plt.subplots(2, 3, figsize=(15,10))
        axs[0,0].imshow(cv2.cvtColor(left_iris_roi, cv2.COLOR_BGR2RGB))
        axs[0,0].set_title("Left Iris ROI")
        axs[0,0].axis("off")
        axs[0,1].imshow(left_pupil_mask, cmap='gray')
        axs[0,1].set_title("Left Pupil Segmentation (Gray)")
        axs[0,1].axis("off")
        axs[0,2].imshow(cv2.cvtColor(left_overlay, cv2.COLOR_BGR2RGB))
        axs[0,2].set_title("Left Pupil Overlay")
        axs[0,2].axis("off")
        axs[1,0].imshow(cv2.cvtColor(right_iris_roi, cv2.COLOR_BGR2RGB))
        axs[1,0].set_title("Right Iris ROI")
        axs[1,0].axis("off")
        axs[1,1].imshow(right_pupil_mask, cmap='gray')
        axs[1,1].set_title("Right Pupil Segmentation (Gray)")
        axs[1,1].axis("off")
        axs[1,2].imshow(cv2.cvtColor(right_overlay, cv2.COLOR_BGR2RGB))
        axs[1,2].set_title("Right Pupil Overlay")
        axs[1,2].axis("off")
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_detailed_angles(left_detail, right_detail, left_h_angle, left_v_angle, right_h_angle, right_v_angle):
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].imshow(cv2.cvtColor(left_detail, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"Left Eye Detail\nAngle: H={left_h_angle:.1f} deg, V={left_v_angle:.1f} deg")
        axs[0].axis("off")
        axs[1].imshow(cv2.cvtColor(right_detail, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Right Eye Detail\nAngle: H={right_h_angle:.1f} deg, V={right_v_angle:.1f} deg")
        axs[1].axis("off")
        plt.tight_layout()
        plt.show()

    def plot_strabismus_analysis_chart(self, left_h_angle, left_v_angle,
                                       right_h_angle, right_v_angle):
        left_diag_text = self.classify_strabismus(left_h_angle, left_v_angle,
                                                  threshold=0.5, eye_side="Left Eye")
        right_diag_text = self.classify_strabismus(right_h_angle, right_v_angle,
                                                   threshold=0.5, eye_side="Right Eye")
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("Strabismus Analiz Karnesi", fontsize=14, fontweight="bold")
        ax.set_xlabel("Horizontal Deviation (°)  [(-) Eso  /  (+) Exo]", fontsize=12)
        ax.set_ylabel("Vertical Deviation (°)  [(-) Hypo /  (+) Hyper]", fontsize=12)
        ax.axhline(y=0, color="black", linewidth=1)
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlim([-15, 15])
        ax.set_ylim([-15, 15])
        ax.plot(left_h_angle, left_v_angle, 'ro', label="Left Eye")
        ax.text(left_h_angle + 0.5, left_v_angle + 0.5,
                left_diag_text, color="red", fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6))
        ax.plot(right_h_angle, right_v_angle, 'bo', label="Right Eye")
        ax.text(right_h_angle + 0.5, right_v_angle + 0.5,
                right_diag_text, color="blue", fontsize=10,
                bbox=dict(facecolor='white', alpha=0.6))
        ax.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    def plot_strabismus_karnesi_dynamic(self,
                                        left_center, left_radius,
                                        cx_left, cy_left,
                                        right_center, right_radius,
                                        cx_right, cy_right,
                                        iris_degree=45,
                                        pupil_degree=15):
        scale_left = pupil_degree / float(left_radius) if left_radius > 0 else 1.0
        scale_right = pupil_degree / float(right_radius) if right_radius > 0 else 1.0

        dx_left = (cx_left - left_center[0]) * scale_left if (cx_left is not None and left_center is not None) else 0
        dy_left = (left_center[1] - cy_left) * scale_left if (cy_left is not None and left_center is not None) else 0
        dx_right = (cx_right - right_center[0]) * scale_right if (cx_right is not None and right_center is not None) else 0
        dy_right = (right_center[1] - cy_right) * scale_right if (cy_right is not None and right_center is not None) else 0
        self._draw_two_eye_diagrams(
            right_ref=(dx_right, dy_right),
            left_ref=(dx_left, dy_left),
            iris_degree=iris_degree,
            pupil_degree=pupil_degree
        )

    def _draw_two_eye_diagrams(self,
                            right_ref=(0,0),
                            left_ref=(0,0),
                            iris_degree=45,
                            pupil_degree=15):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        major_angles = [0, 15, 30, 45, 90, 135, 180, 225, 270, 315, 360]
        region_lines = [0, 90, 180, 270]
        def polar_to_cartesian(deg, r):
            rad = math.radians(deg)
            return (r * math.cos(rad), r * math.sin(rad))
        def draw_eye(ax, eye_name, ref_xy):
            ax.set_aspect("equal")
            ax.set_xlim(-iris_degree-5, iris_degree+5)
            ax.set_ylim(-iris_degree-5, iris_degree+5)
            ax.set_title(eye_name, fontsize=12, fontweight='bold')
            ax.axhline(0, color="black", linewidth=1)
            ax.axvline(0, color="black", linewidth=1)
            iris_circle = Circle((0, 0),
                                iris_degree,
                                edgecolor="blue",
                                facecolor="#444444",
                                alpha=0.3,
                                lw=2)
            ax.add_patch(iris_circle)
            pupil_circle = Circle((0, 0),
                                pupil_degree,
                                edgecolor="black",
                                facecolor="black",
                                alpha=0.8,
                                lw=1)
            ax.add_patch(pupil_circle)
            for angle in region_lines:
                x2, y2 = polar_to_cartesian(angle, iris_degree)
                ax.plot([0, x2], [0, y2], color="white", linestyle="--", linewidth=1)
            for ang in major_angles:
                if ang == 360:
                    continue
                px, py = polar_to_cartesian(ang, iris_degree)
                ax.plot(px, py, 'go', markersize=5)
                ax.text(px * 1.06, py * 1.06, f"{ang}°",
                        color="yellow", fontsize=8,
                        ha="center", va="center")
            ax.text(iris_degree + 3, 0, "Exotropia →", color="red", ha="left", va="center")
            ax.text(-iris_degree - 3, 0, "← Esotropia", color="red", ha="right", va="center")
            ax.text(0, iris_degree + 3, "Hypertropia ↑", color="red", ha="center", va="bottom")
            ax.text(0, -iris_degree - 3, "↓ Hypotropia", color="red", ha="center", va="top")
            ax.text(iris_degree / 2, iris_degree / 2, "region 1", color="cyan")
            ax.text(-iris_degree / 2, iris_degree / 2, "region 2", color="cyan")
            ax.text(-iris_degree / 2, -iris_degree / 2, "region 3", color="cyan")
            ax.text(iris_degree / 2, -iris_degree / 2, "region 4", color="cyan")
            rx, ry = ref_xy
            ax.plot(rx, ry, 'r*', markersize=12, label="Reflected Light")
            ax.text(rx + 2, ry + 2, f"({rx:.1f}, {ry:.1f})", color="white",
                    bbox=dict(facecolor="red", alpha=0.6),
                    fontsize=9)
            ax.legend(loc="lower right")
            ax.grid(False)
        draw_eye(axes[0], "Right Eye", right_ref)
        draw_eye(axes[1], "Left Eye", left_ref)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Örnek test: örnek açı değerleri ile karne çizimi
    left_h_angle = -5.0
    left_v_angle = 6.0
    right_h_angle = 4.0
    right_v_angle = -4.0
    dummy_detector = None  # Gerçek iris_detector eklenirse kullanılabilir.
    analyzer = IrisAnalyzer(dummy_detector)
    analyzer.plot_strabismus_analysis_chart(left_h_angle, left_v_angle, right_h_angle, right_v_angle)
