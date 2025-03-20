import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, generate_binary_structure
from skimage import morphology

# Função para calcular o limiar ideal para binarização usando o método de Otsu
def calcular_limiar_otsu(image_array):
    # Calcula o histograma da imagem e o total de pixels
    histograma, _ = np.histogram(image_array, bins=256, range=(0, 256))
    total_pixels = image_array.size
    soma_total = np.sum(np.arange(256) * histograma)
    
    # Inicializa variáveis para encontrar o melhor limiar
    melhor_limiar = 0
    variancia_maxima = 0
    soma_fundo = 0
    peso_fundo = 0
    
    # Itera sobre todos os possíveis limiares
    for t in range(256):
        peso_fundo += histograma[t]
        if peso_fundo == 0:
            continue
        
        peso_frente = total_pixels - peso_fundo
        if peso_frente == 0:
            break
        
        # Calcula as médias e a variância entre classes
        soma_fundo += t * histograma[t]
        media_fundo = soma_fundo / peso_fundo
        media_frente = (soma_total - soma_fundo) / peso_frente
        
        variancia_entre_classes = peso_fundo * peso_frente * (media_fundo - media_frente) ** 2
        
        # Atualiza o melhor limiar se a variância for maior
        if variancia_entre_classes > variancia_maxima:
            variancia_maxima = variancia_entre_classes
            melhor_limiar = t
    
    # Ajuste do limiar para torná-lo mais restritivo
    return melhor_limiar * 1.1  # Aumenta o limiar em 10%

# Função para binarizar a imagem e aplicar operações morfológicas
def binarizar_imagem(image_array, limiar):
    # Binariza a imagem com base no limiar
    imagem_binaria = image_array < limiar
    
    # Aplica abertura morfológica para remover pequenos objetos
    imagem_binaria = morphology.binary_opening(imagem_binaria, morphology.square(3))
    
    # Aplica fechamento morfológico para fechar buracos
    imagem_binaria = morphology.binary_closing(imagem_binaria, morphology.square(5))
    
    # Retorna a imagem binarizada como uint8
    return imagem_binaria.astype(np.uint8)

# Função para identificar componentes conectados na imagem binária
def identificar_componentes(imagem_binaria, area_minima=200, proporcao_maxima=2.0):
    # Define a estrutura de conexão para os componentes
    estrutura_conexao = generate_binary_structure(2, 2)
    
    # Rotula os componentes conectados na imagem
    rotulos, total_objetos = label(imagem_binaria, structure=estrutura_conexao)
    
    # Lista para armazenar informações dos objetos detectados
    lista_objetos = []
    for id_objeto in range(1, total_objetos + 1):
        # Obtém as coordenadas dos pixels do objeto
        coordenadas = np.argwhere(rotulos == id_objeto)
        
        # Calcula os limites do objeto
        min_linha, min_coluna = coordenadas.min(axis=0)
        max_linha, max_coluna = coordenadas.max(axis=0)
        
        # Calcula a área do objeto
        area = coordenadas.shape[0]
        largura = max_coluna - min_coluna
        altura = max_linha - min_linha
        proporcao = largura / altura if altura != 0 else 0
        
        # Filtra por área mínima e proporção máxima
        if area > area_minima and (proporcao <= proporcao_maxima and proporcao >= 1/proporcao_maxima):
            lista_objetos.append({
                'posicao': (min_coluna, min_linha),
                'dimensoes': (largura, altura),
                'area': area
            })
    
    # Retorna os rótulos e a lista de objetos detectados
    return rotulos, lista_objetos

# Função para desenhar retângulos ao redor dos objetos detectados
def marcar_objetos(imagem_caminho, lista_objetos):
    # Abre a imagem e a converte para RGB
    imagem = Image.open(imagem_caminho).convert("RGB")
    
    # Cria um objeto de desenho para a imagem
    desenho = ImageDraw.Draw(imagem)
    
    # Itera sobre os objetos detectados
    for objeto in lista_objetos:
        # Obtém as coordenadas e dimensões do objeto
        min_coluna, min_linha = objeto['posicao']
        largura, altura = objeto['dimensoes']
        
        # Desenha um retângulo ao redor do objeto
        desenho.rectangle([min_coluna, min_linha, min_coluna + largura, min_linha + altura], outline="red", width=2)
    
    # Retorna a imagem com os objetos marcados
    return imagem

# Função principal para contar objetos na imagem
def contar_objetos(imagem):
    # Calcula o limiar ideal para binarização usando o método de Otsu
    limiar = calcular_limiar_otsu(imagem)
    
    # Binariza a imagem com base no limiar calculado e aplica operações morfológicas
    imagem_binarizada = binarizar_imagem(imagem, limiar)
    
    # Identifica componentes conectados na imagem binária e retorna os objetos detectados
    rotulos, objetos = identificar_componentes(imagem_binarizada)
    
    # Retorna a imagem binarizada e a lista de objetos detectados
    return imagem_binarizada, objetos