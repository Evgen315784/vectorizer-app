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
        <input type="file" name="image"><br><br>
        <label for="mode">Выберите режим:</label>
        <select name="mode">
            <option value="basic">Обычный контур</option>
            <option value="smooth">Сглаженный контур</option>
            <option value="minimal">Минималистично</option>
        </select><br><br>
        <input type="submit" value="Векторизовать">
    </form>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def vectorize():
    if request.method == 'POST':
        file = request.files['image']
        mode = request.form.get('mode', 'basic')
        image = Image.open(file.stream).convert('RGB')
        image_np = np.array(image)

        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.svg') as tmp_svg:
            dwg = svgwrite.Drawing(tmp_svg.name, profile='tiny')

            for contour in contours:
                if len(contour) < 20:
                    continue  # Пропускаем мелкий мусор

                if mode == 'smooth':
                    epsilon = 2.0
                    contour = cv2.approxPolyDP(contour, epsilon, True)
                elif mode == 'minimal':
                    epsilon = 5.0
                    contour = cv2.approxPolyDP(contour, epsilon, True)

                points = [(int(x), int(y)) for [[x, y]] in contour]
                dwg.add(dwg.polyline(points=points, stroke='black', fill='none', stroke_width=0.8))

            dwg.save()
            return send_file(tmp_svg.name, mimetype='image/svg+xml', as_attachment=True, download_name='vectorized.svg')

    return render_template_string(HTML_FORM)

if __name__ == '__main__':
    app.run(debug=True)
