#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import tensorflow as tf


class IrisLandmark:
    def __init__(self, model_path='iris_landmark/iris_landmark.tflite', num_threads=1):
        # TensorFlow Lite yorumlayıcısını (interpreter) oluştur ve bellek ayır.
        self._interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def __call__(self, image):
        """
        Verilen görüntüyü modele gönderir, gerekli önişlemeleri yapar ve
        model çıktısını (eye_contour ve iris landmark değerleri) döndürür.
        """
        input_shape = self._input_details[0]['shape']  # Örneğin: [1, height, width, channels]
        target_height, target_width = input_shape[1], input_shape[2]

        # 1. Görüntüyü BGR'den RGB'ye çevir ve float32 olarak normalize et (0-1 arası)
        img_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # 2. Yeniden boyutlandırma: OpenCV'nin cv.resize fonksiyonu genellikle tf.image.resize'den daha hızlıdır.
        resized_img = cv.resize(img_rgb, (target_width, target_height), interpolation=cv.INTER_CUBIC)

        # 3. Normalizasyon: Modelin beklediği şekilde (-1, 1) aralığına taşımak için
        normalized_img = (resized_img - 0.5) / 0.5

        # 4. Batch boyutu ekle (1, height, width, channels)
        input_tensor = np.expand_dims(normalized_img, axis=0)

        # 5. Modelin girişine tensor'ü yerleştir ve inferansı çalıştır
        self._interpreter.set_tensor(self._input_details[0]['index'], input_tensor)
        self._interpreter.invoke()

        # 6. Çıktıları al
        eye_contour = self._interpreter.get_tensor(self._output_details[0]['index'])
        iris = self._interpreter.get_tensor(self._output_details[1]['index'])

        # Squeeze ile gereksiz boyutları kaldırıyoruz.
        return np.squeeze(eye_contour), np.squeeze(iris)

    def get_input_shape(self):
        """Modelin beklediği (width, height) boyutunu döndürür."""
        input_shape = self._input_details[0]['shape']
        return [input_shape[2], input_shape[1]]  # [width, height]


# Örnek kullanım
if __name__ == '__main__':
    # Model nesnesini oluştur
    iris_detector = IrisLandmark()

    # Test için bir görüntü yükleyin (dosya yolunu kendinize göre ayarlayın)
    image_path = "test_image.jpg"
    image = cv.imread(image_path)
    if image is None:
        print("Görüntü okunamadı!")
        exit()

    # Modelin beklediği giriş boyutunu yazdırın
    print("Modelin beklediği giriş boyutu:", iris_detector.get_input_shape())
    print("Görüntü boyutu:", image.shape)

    # Modeli çalıştır ve çıktıları al
    eye_contour, iris = iris_detector(image)

    # Çıktıları konsola yazdırın
    print("Eye Contour Çıkışı:", eye_contour)
    print("İris Çıkışı:", iris)
