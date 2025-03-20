import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.ndimage import label, binary_closing, generate_binary_structure
from skimage import morphology


def otsu_threshold(image_array):
    """Aplica o algoritmo de Otsu para encontrar o melhor limiar de binarização."""
    hist = np.histogram(image_array.flatten(), bins=256, range=(0, 256))[0]
    total_pixels = image_array.size
    sum_total = np.sum(np.arange(256) * hist)
    
    max_variance = 0
    best_threshold = 0
    sum_background = 0
    weight_background = 0
    
    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
        
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        
        sum_background += i * hist[i]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance_between > max_variance:
            max_variance = variance_between
            best_threshold = i
    
    return best_threshold

def segmentar_imagem(image_array, threshold):
    """Segmenta a imagem usando o limiar de Otsu e aplica fechamento morfológico."""
    binary_image = image_array < threshold  # Inverte a imagem para fundo preto e objetos brancos
    binary_image = morphology.closing(binary_image, morphology.square(3))  # Remove pequenos buracos
    return binary_image.astype(np.uint8)

def detectar_objetos(binary_image, min_area=100):
    """Identifica objetos na imagem usando marcadores e rotulação de componentes conectados."""
    estrutura = generate_binary_structure(2, 2)
    labeled, num_objects = label(binary_image, structure=estrutura)
    
    objetos = []
    for obj_id in range(1, num_objects + 1):
        posicoes = np.argwhere(labeled == obj_id)
        minr, minc = posicoes.min(axis=0)
        maxr, maxc = posicoes.max(axis=0)
        area = len(posicoes)
        
        if area > min_area:
            objetos.append({
                'posicao': (minc, minr),
                'dimensoes': (maxc - minc, maxr - minr),
                'area': area
            })
    
    return labeled, objetos

def desenhar_retangulos(image_path, objetos):
    """Desenha retângulos ao redor dos objetos detectados."""
    imagem = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(imagem)
    
    for obj in objetos:
        minc, minr = obj['posicao']
        largura, altura = obj['dimensoes']
        draw.rectangle([minc, minr, minc + largura, minr + altura], outline="red", width=2)
    
    return imagem

def contagem_de_objetos(imagem):
    
    threshold = otsu_threshold(imagem)
    imagem_binaria = segmentar_imagem(imagem, threshold)
    objetos_detectados, objetos = detectar_objetos(imagem_binaria)

    return imagem_binaria, objetos