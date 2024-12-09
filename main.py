from flask import Flask, request, jsonify  
from flask_cors import CORS  
from tensorflow.keras.models import load_model  
from tensorflow.keras.utils import img_to_array  
from PIL import Image  
import numpy as np  
from io import BytesIO  

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Aktifkan CORS agar API bisa diakses dari berbagai domain
CORS(app)

# Load model yang sudah dilatih
try:
    model = load_model("model_guntingbatukertas.keras")  # Memuat model .keras
except Exception as e:
    raise ValueError(f"Error loading model: {str(e)}")

LABELS = ['paper', 'rock', 'scissors']

# Route untuk halaman utama API
@app.route('/')
def welcome():
    return jsonify({"message": "Selamat Datang di API Model Gambar Permainan Tangan Gunting, Batu dan Kertas"}), 200  # Response untuk halaman utama

# Route untuk prediksi gambar
@app.route('/predict', methods=['POST'])
def predict():
    # Pastikan ada file yang diunggah melalui request
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']  # Mengambil file dari request

    try:
        # Preprocessing gambar
        image = Image.open(BytesIO(file.read()))  # Membaca file sebagai stream gambar
        image = image.resize((160, 160))         # Mengubah ukuran gambar sesuai input model
        image = img_to_array(image)              # Konversi gambar ke array numpy
        image = np.expand_dims(image, axis=0)    # Tambahkan dimensi batch untuk input model
        image = image / 255.0                    # Normalisasi piksel gambar ke rentang [0, 1]

        # Prediksi menggunakan model
        prediction = model.predict(image)  # Menghasilkan probabilitas untuk setiap kelas
        predicted_class = LABELS[np.argmax(prediction)]  # Mengambil label dengan probabilitas tertinggi
        confidence = float(np.max(prediction))  

        # Response hasil prediksi
        return jsonify({
            "prediction": predicted_class,  
            "confidence": confidence        
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)  
