from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from mainFlask import analyze_image  # your analysis function
from pdf_report_generator import PDFReportGenerator  # your report generator module

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Check if the user requested a PDF report
            generate_report_flag = request.form.get("generate_report") == "yes"
            
            # Run your analysis function (make sure it works in web mode)
            results = analyze_image(filepath, display_in_tk=False, segmentation_method="YOLO")
            
            # Save the debug image output (or any representative image) to the uploads folder
            debug_filename = "debug_" + filename
            debug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
            # Assuming results["debug_image"] is a valid OpenCV image:
            import cv2
            cv2.imwrite(debug_filepath, results["debug_image"])
            
            # If report generation was requested:
            report_filename = None
            if generate_report_flag:
                pdf_report_path = os.path.join(app.config['UPLOAD_FOLDER'], "report_" + filename.split('.')[0] + ".pdf")
                report_generator = PDFReportGenerator()
                report_generator.generate_report(results, output_path=pdf_report_path)
                report_filename = os.path.basename(pdf_report_path)
            
            # Redirect to a results page with both image and (optionally) report
            return redirect(url_for('result', debug_filename=debug_filename, report_filename=report_filename))
    return render_template('index.html')

@app.route('/result')
def result():
    debug_filename = request.args.get("debug_filename")
    report_filename = request.args.get("report_filename")
    return render_template('result.html', debug_filename=debug_filename, report_filename=report_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
