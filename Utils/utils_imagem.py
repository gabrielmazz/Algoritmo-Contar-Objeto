import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console
from rich.prompt import Prompt
import os
import cv2
from PIL import Image, ImageDraw
from io import BytesIO
import requests

# Leitura da imagem
def leitura_Imagem(nome):
    
    imagem = Image.open(nome).convert('L')
    return np.array(imagem, dtype=np.uint8)

# Realiza a plotagem das imagens com o matplotlib
def plotagem_imagem(Imagem_Original, Imagem_Resultado, Imagem_Resultado_Retangulos):
    
    # Cria a figura com os subplots
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    
    # Adiciona as imagens nos subplots
    axs[0].imshow(Imagem_Original, cmap='Greys')
    axs[0].set_title('Imagem Original')
    
    axs[1].imshow(Imagem_Resultado, cmap='Greys')
    axs[1].set_title('Imagem Resultado')
    
    axs[2].imshow(Imagem_Resultado_Retangulos)
    axs[2].set_title('Imagem Resultado com Retângulos')
    
    # Remove os eixos dos subplots
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])
    
    # Mostra a figura com os subplots
    plt.show()
    
def desenhar_retangulos(imagem_path, objetos):
    
    # Converte a imagem para RGB
    imagem = Image.open(imagem_path).convert('RGB')
    
    # Cria o objeto para desenhar na imagem
    draw = ImageDraw.Draw(imagem)
    
    # Desenha os retângulos
    for obj in objetos:
        draw.rectangle([obj['posicao'], (obj['posicao'][0] + obj['dimensoes'][0], obj['posicao'][1] + obj['dimensoes'][1])], outline='red', width=3)
        
    return imagem
    
def salvar_imagem(Imagem_Binaria, nome):
    
    plt.imsave(nome, Imagem_Binaria, cmap='Greys')
    
def lista_imagens_pasta(pasta, console):
    
    # Lista as imagens disponíveis na pasta
    imagens = [f for f in os.listdir(pasta)]
    
    # Printa as imagens
    for i, imagem in enumerate(imagens):
        console.print('{}. {}'.format(i+1, imagem))
        
    return imagens

def escolher_imagens(imagens, console):
    
    # Escolhe uma imagem para aplicar o filtro Box
    while True:
        escolha = int(Prompt.ask('Escolha uma imagem para aplicar a [bold purple]Contagem de Objeto[/bold purple]', console=console))
        
        if escolha > 0 and escolha <= len(imagens):
            return imagens[escolha-1]
        else:
            console.print('Escolha inválida. Tente novamente.')
            
def download_imagem(args):
    
    console = Console()
    
    # Baixa a imagem da URL
    response = requests.get(args.url)
    
    # Verifica se a requisição foi bem sucedida
    if response.status_code == 200:
        # Lê a imagem
        Imagem = Image.open(BytesIO(response.content))
        
        # Define o nome da imagem
        nome_imagem = "IMAGEM_BAIXADA_URL"  # Nome fixo
        extensao = args.url.split('.')[-1]  # Extrai a extensão da URL (ex: jpg, png, etc.)
        
        # Salva a imagem com o novo nome
        Imagem.save(f'./imagens/{nome_imagem}.{extensao}')
        
    else:
        console.print('Erro ao baixar a imagem. Tente novamente.')

def deletar_imagem(nome):
    
    # Deleta a imagem
    os.remove('./imagens/{}'.format(nome))
