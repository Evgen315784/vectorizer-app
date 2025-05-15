from flask import Flask, request, send_file, render_template_string
import cv2
import numpy as np
import svgwrite
import tempfile
from skimage import measure
from PIL import Image

app = Flask(__name__)

HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Векторизатор</title>
</head>
<body>
    <h2>Загрузите изображение для векторизации:</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit" value="Векторизовать">
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def vectorize():
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours = measure.find_contours(thresh, 0.8)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.svg') as tmp_svg:
            dwg = svgwrite.Drawing(tmp_svg.name, profile='tiny')

            for contour in contours:
                points = [(x, y) for y, x in contour]
                dwg.add(dwg.polyline(points=points, stroke='black', fill='none', stroke_width=0.5))

            dwg.save()
            return send_file(tmp_svg.name, mimetype='image/svg+xml', as_attachment=True, download_name='vectorized.svg')

    return render_template_string(HTML_FORM)

if __name__ == '__main__':
    app.run(debug=True)
