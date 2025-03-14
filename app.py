from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from predict import predict_tumor

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = os.path.join(UPLOAD_FOLDER, secure_filename(f.filename))
        f.save(file_path)
        
        result = predict_tumor(file_path)
        return render_template('import.html', result=result, image=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
