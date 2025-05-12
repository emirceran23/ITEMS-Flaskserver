from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime
import os
from PIL import Image as PILImage

pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

class PDFReportGenerator:
    def __init__(self):
        self.page_size = A4
        self.styles = getSampleStyleSheet()
        
        for style in self.styles.byName.values():
            style.fontName = 'DejaVuSans'
        self.styles.add(ParagraphStyle(name='Justify', alignment=4, leading=16, fontName='DejaVuSans'))
    
    def get_proportional_image(self, image_path, max_width, max_height):
        try:
            pil_img = PILImage.open(image_path)
            img_width, img_height = pil_img.size
            width_scale = max_width / img_width
            height_scale = max_height / img_height
            scale = min(width_scale, height_scale)
            new_width = img_width * scale
            new_height = img_height * scale
            return Image(image_path, width=new_width, height=new_height)
        except Exception as e:
            print(f"Resim {image_path} yüklenirken hata: {e}")
            return None

    def generate_report(self, results, output_path="analysis_report.pdf"):
        doc = SimpleDocTemplate(output_path, pagesize=self.page_size, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        story = []
        
        title = Paragraph("Detaylı Analiz Raporu", self.styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Rapor Tarihi: {now}", self.styles['Normal']))
        story.append(Spacer(1, 24))
        
        seg_info = (
            f"<b>Segmentasyon Yöntemi:</b> {results.get('segmentation_method', 'Bilinmiyor')}<br/>"
            f"<b>Iris Segmentasyonu:</b> {results.get('iris_segmentation', 'Bilinmiyor')}<br/>"
            f"<b>Pupil Segmentasyonu:</b> {results.get('pupil_segmentation', 'Bilinmiyor')}<br/>"
            f"<b>Yansıma Segmentasyonu:</b> {results.get('reflection_segmentation', 'Bilinmiyor')}"
        )
        story.append(Paragraph("<b>Segmentasyon Raporu</b>", self.styles['Heading2']))
        story.append(Paragraph(seg_info, self.styles['Normal']))
        story.append(Spacer(1, 24))

        story.append(Paragraph("<b>Kamera Özellikleri</b>", self.styles['Heading2']))
        camera_info = (
            f"<b>Odak Uzaklığı:</b> {results.get('focal_length', 'Bilinmiyor')}<br/>"
            f"<b>Diyafram Değeri:</b> {results.get('aperture', 'Bilinmiyor')}<br/>"
            f"<b>Sensör Genişliği:</b> {results.get('sensor_width', 'Bilinmiyor')}<br/>"
            f"<b>Kamera Markası:</b> {results.get('camera_brand', 'Bilinmiyor')}<br/>"
            f"<b>Kamera Modeli:</b> {results.get('camera_model', 'Bilinmiyor')}<br/>"
            f"<b>Exposure Time</b> {results.get('exposure_time', 'Bilinmiyor')}<br/>"
            f"<b>ISO Değeri:</b> {results.get('iso_value', 'Bilinmiyor')}"
        )
        story.append(Paragraph(camera_info, self.styles['Normal']))
        story.append(Spacer(1, 24))

        
        image_descriptions = {
            "debug_image": "Bu görüntüde iris, pupil ve refleksiyon noktalarının segmentasyonu gerçekleştirilmiştir. Göz hizalanma durumu pupil ve refleksiyon noktalarının ilişkisi analiz edilerek değerlendirilmiştir.",
            "strabismus_chart_image": "Strabismus karnesi, göz hizalanma bozukluklarını belirlemek için kullanılan bir haritalama yöntemidir. İris ve pupil merkezleri ile refleksiyon noktalarının koordinatları incelenmiştir.",
            "detailed_analysis_image": "İris, pupil ve refleksiyon noktalarının belirginliği segmentasyon algoritmaları ile çıkarılmıştır. Pupillerin refleksiyon noktalarına olan uzaklıkları değerlendirilerek göz kayması analiz edilmiştir.",
            "centers_analysis_image": "Refleksiyon noktalarının pupillerle olan hizası analiz edilmiştir. Göz kaslarındaki olası dengesizlikler refleksiyon noktalarının asimetrik dağılımıyla tespit edilmiştir.",
            "roi_angles_image": "Göz bölgesindeki referans noktalarla açısal analiz gerçekleştirilmiştir. Pupillerin konumu ve göz eksenleri arasındaki açısal farklar klinik değerlendirme için raporlanmıştır.",
            "deviation_polar_left": "Sol göz için pupil ve refleksiyon merkezi arasındaki açısal sapma polar koordinat sisteminde haritalandırılmıştır. Bu grafik göz ekseni sapmalarını gösterir.",
            "deviation_polar_right": "Sağ göz için pupil ve refleksiyon merkezi arasındaki açısal sapma polar koordinat sisteminde haritalandırılmıştır. Göz hizalama bozukluklarının tespiti için kritik öneme sahiptir."
        }
        
        for title_text, key in image_descriptions.items():
            path = results.get(title_text)
            if path is not None and os.path.exists(path):
                story.append(Paragraph(f"<b>{title_text.replace('_', ' ').title()}</b>", self.styles['Heading2']))
                img_flowable = self.get_proportional_image(path, 5*inch, 4*inch)
                if img_flowable is not None:
                    story.append(img_flowable)
                
                story.append(Paragraph(image_descriptions[title_text], self.styles['Justify']))
                story.append(Spacer(1, 12))
            else:
                story.append(Paragraph(f"{title_text.replace('_', ' ').title()} bulunamadı.", self.styles['Normal']))
                story.append(Spacer(1, 12))

        
        import math

        pinhole_results = results.get('pinhole_strabismus_results')
        left_h_angle = pinhole_results.get('left_h_angle_deg')
        left_v_angle = pinhole_results.get('left_v_angle_deg')
        right_h_angle = pinhole_results.get('right_h_angle_deg')
        right_v_angle = pinhole_results.get('right_v_angle_deg')

        left_angle = math.sqrt(left_h_angle**2 + left_v_angle**2)
        right_angle = math.sqrt(right_h_angle**2 + right_v_angle**2)

        result_values = (
            f"<b>Gerçek Pupiller Arası mesafe (mm) :</b> {pinhole_results.get('real_ipd_mm', 'Bilinmiyor')}<br/>"
            f"<b>Sol Göz Şaşılık Açısı (°) :</b> {left_angle}<br/>",
            f"<b>Sol Göz Şaşılık Sınıflandırması :</b> {pinhole_results.get('left_classification', 'Bilinmiyor')}<br/>",
            f"<b>Sol Göz Sapma Büyüklüğü (mm)) :</b> {pinhole_results.get('left_deviation_magnitude', 'Bilinmiyor')}<br/>",
            f"<b>Sol Göz Sapma Büyüklüğü (PD):</b> {22 * pinhole_results.get('left_deviation_magnitude', 'Bilinmiyor')}<br/>",
            f"<b>Sağ Göz Şaşılık Açısı (°) :</b> {right_angle}<br/>",
            f"<b>Sağ Göz Şaşılık Sınıflandırması :</b> {pinhole_results.get('right_classification', 'Bilinmiyor')}<br/>",
            f"<b>Sağ Göz Sapma Büyüklüğü (mm)) :</b> {pinhole_results.get('right_deviation_magnitude', 'Bilinmiyor')}<br/>",
            f"<b>Sağ Göz Sapma Büyüklüğü (PD):</b> {22 * pinhole_results.get('right_deviation_magnitude', 'Bilinmiyor')}<br/>"
        )

        story.append(Paragraph("<b>Analiz Sonuçları</b>", self.styles['Heading2']))
        result_str = "\n".join(result_values)
        story.append(Paragraph(result_str, self.styles['Normal']))
        story.append(Spacer(1, 24))


        
        conclusion_text = (
            "Strabismus analizi kapsamında gerçekleştirilen ölçümler ve görüntüleme verileri klinik olarak değerlendirilmiştir. "
            "Elde edilen sonuçlar göz hizalama bozukluklarını belirlemek ve klinik teşhis süreçlerine destek olmak amacıyla kullanılabilir. "
            "Detaylı göz muayenesi ve ek tanı testleri ile birlikte daha kesin bir değerlendirme yapılması önerilmektedir."
        )
        
        story.append(PageBreak())
        story.append(Paragraph("<b>Sonuç ve Genel Yorumlar</b>", self.styles['Heading2']))
        story.append(Paragraph(conclusion_text, self.styles['Justify']))
        story.append(Spacer(1, 12))
        
        doc.build(story)
        print(f"Rapor başarıyla oluşturuldu: {output_path}")
