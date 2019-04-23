from app import app
from flask import render_template, url_for, request,redirect,send_file
import io
from PIL import Image
import cv2
import numpy as np
from app import test_extract_feature

@app.route('/')
def index():
    
    return render_template('index.html',title="test")


@app.route('/store',methods = ['POST'])
def store_data():
    if request.method == 'POST':
        print('test')
        fl = request.files['gambar']
        in_memory_file = io.BytesIO()
        fl.save(in_memory_file)
        data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
        color_image_flag = 0
        img = cv2.imdecode(data, color_image_flag)

        img = test_extract_feature.main(img)
    
        img = cv2.imdecode(data, color_image_flag)

        img.save("your_file.png")

    

        return send_file(data, mimetype='image/BMP', as_attachment=False)
        # return render_template('preprocessing.html', img_preprocess = img)

    else:
        print('get')
        return redirect(url_for('/')) 