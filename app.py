from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from flask_socketio import SocketIO
import os
from werkzeug.utils import secure_filename
import cv2
from mainFlask import analyze_image
from pdf_report_generator import PDFReportGenerator

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['SECRET_KEY'] = 'secret!'

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    return "Server is running."

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Run analysis
        results = analyze_image(filepath, display_in_tk=False, segmentation_method="YOLO")

        # Save debug image
        debug_filename = "debug_" + filename
        debug_filepath = os.path.join(app.config['UPLOAD_FOLDER'], debug_filename)
        cv2.imwrite(debug_filepath, results["debug_image"])

        # Generate PDF report
        pdf_filename = "report_" + os.path.splitext(filename)[0] + ".pdf"
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
        report_generator = PDFReportGenerator()
        report_generator.generate_report(results, output_path=pdf_path)

        # Return URLs
        debug_url = url_for('uploaded_file', filename=debug_filename, _external=True)
        pdf_url = url_for('uploaded_file', filename=pdf_filename, _external=True)
        return jsonify({
            'debug_image_url': debug_url,
            'pdf_url': pdf_url
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)
