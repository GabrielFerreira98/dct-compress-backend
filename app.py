from flask import Flask
from flask_cors import CORS

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import cv2
import numpy as np
from skimage import io
from skimage.metrics import structural_similarity as ssim
import pywt
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

# Função da Máscara

def z_scan_mask(C, N):
    mask = np.zeros((N, N))
    start = 0
    mask_m = start
    mask_n = start
    for i in range(C):
        if i == 0:
            mask[mask_m, mask_n] = 1
        else:
            # If even, move upward to the right
            if (mask_m + mask_n) % 2 == 0:
                mask_m -= 1
                mask_n += 1
                # If it exceeds the upper boundary, move downward
                if mask_m < 0:
                    mask_m += 1
                # If it exceeds the right boundary, move left
                if mask_n >= N:
                    mask_n -= 1
            # If odd, move downward to the left
            else:
                mask_m += 1
                mask_n -= 1
                # If it exceeds the lower boundary, move upward
                if mask_m >= N:
                    mask_m -= 1
                # If it exceeds the left boundary, move right
                if mask_n < 0:
                    mask_n += 1
            mask[mask_m, mask_n] = 1
    return mask

# overlaying the mask, discarding the high-frequency components
def Compress(img, mask, N):
    img_dct = np.zeros((img.shape[0] // N * N, img.shape[1] // N * N), dtype=np.float32)
    for m in range(0, img_dct.shape[0], N):
        for n in range(0, img_dct.shape[1], N):
            block = img[m:m+N, n:n+N]
            # DCT
            coeff = cv2.dct(block)
            # IDCT, but only the parts of the image where the mask has a value of 1 are retained
            iblock = cv2.idct(coeff * mask)
            img_dct[m:m+N, n:n+N] = iblock
    return img_dct

# Function for MSE
def mean_squared_error(image1, image2):
    error = np.sum((image1.astype('float') - image2.astype('float'))**2)
    error = error/float(image1.shape[0] * image2.shape[1])
    return error

def image_comparison(image1, image2):
    # input image must have the same dimension for comparison
    image2 = cv2.resize(image2,(image1.shape[1::-1]),interpolation=cv2.INTER_AREA)
    m = mean_squared_error(image1, image2)
    s = ssim(image1, image2, data_range=image1.max() - image1.min(), multichannel=True)
    psnr = cv2.PSNR(image1, image2)
    return m, s, psnr

@app.route('/')
def index():
    return {'Hello, World!'}

@app.route('/calculate', methods=['POST'])
def calculate_image():
    try:
        data = request.get_json()
        image_path = data.get('image')
        amount_of_coeffs = data.get('amount_of_coeffs')
        block_size = data.get('block_size')

        if not image_path:
            return jsonify({'error': 'Caminho da imagem não encontrado'}), 400

        # Carregar a imagem utilizando OpenCV
        image = cv2.imread(image_path)

        if image is None:
            return jsonify({'error': 'Não foi possível carregar a imagem'}), 400

        # Operação de Compressão

        img = io.imread(image_path, as_gray=True).astype(np.float32)

        mask = z_scan_mask(amount_of_coeffs, block_size)
        compressed_img = Compress(img, mask, block_size)

        # Calculo das Métricas
        mse, ssim_value, psnr_value = image_comparison(img, compressed_img)
        print(mse, ssim_value, psnr_value)

        # Salvar a imagem temporária para servir como resposta
        cv2.imwrite('compressed_image.jpg', compressed_img)

        # Enviar a imagem comprimida de volta para o frontend
        return send_file('compressed_image.jpg', mimetype='image/jpeg')

    except Exception as e:
        app.logger.error(f'Erro: {e}')
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_metrics', methods=['POST'])
def get_metrics():
    try:
        data = request.get_json()
        image_path = data.get('image')
        amount_of_coeffs = data.get('amount_of_coeffs')
        block_size = data.get('block_size')

        if not image_path:
            return jsonify({'error': 'Caminho da imagem não encontrado'}), 400

        # Carregar a imagem utilizando OpenCV
        image = cv2.imread(image_path)

        if image is None:
            return jsonify({'error': 'Não foi possível carregar a imagem'}), 400

        # Operação de Compressão

        img = io.imread(image_path, as_gray=True).astype(np.float32)

        mask = z_scan_mask(amount_of_coeffs, block_size)
        compressed_img = Compress(img, mask, block_size)

        # Calculo das Métricas
        mse, ssim_value, psnr_value = image_comparison(img, compressed_img)
        print(mse, ssim_value, psnr_value)

        response = {
            'mse': mse,
            'ssim': ssim_value,
            'psnr': psnr_value
        }

        return jsonify(response), 200
    
    except Exception as e:
        app.logger.error(f'Erro: {e}')
        return jsonify({'error': str(e)}), 500
    
# if __name__ == '__main__':
#     app.run(debug=True,port=5000)

# CORRETO
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)