import matplotlib.pyplot as plt
import argparse
from rich.console import Console
from rich.prompt import Prompt
from rich.progress import Progress
import Contagem.contagem as cont_obj
import Utils.utils_imagem as ut_img
import Utils.utils_code as ut_code
import time
import Utils.library_checker as lib_checker

# Variáveis para passagem de argumentos via terminal
parser = argparse.ArgumentParser()

# Argumento para salvar a imagem na pasta de resultados
SAVE = parser.add_argument('--save', action='store_true', help='Salvar a imagem na pasta de resultados')
INFO = parser.add_argument('--info', action='store_true', help='Exibir o tempo de execução')
URL_IMAGE = parser.add_argument('--url', type=str, help='URL da imagem para usar no algoritmo')

def contador_de_objetos(imagem_escolhida, tipo):
    
    # Inicializa o tempo de execução
    start_time = time.time()
    
    with Progress() as progress:
        
        # Atualiza o progresso
        task = progress.add_task("[cyan]Processando...", total=4) 

        time.sleep(1)
        
        # Leitura da imagem
        progress.update(task, advance=1, description='[cyan]Lendo a imagem...')
        Imagem_Original = ut_img.leitura_Imagem('./imagens/{}'.format(imagem_escolhida))    

        time.sleep(1)

        # Realiza a contagem de objetos
        progress.update(task, advance=1, description='[cyan]Contando os objetos...')
        imagem_resultado, contagem_objetos = cont_obj.contar_objetos(Imagem_Original)
        
        time.sleep(1)
        
        # Desenhando os objetos na imagem
        progress.update(task, advance=1, description='[cyan]Desenhando os retângulos na imagem...')
        imagem_resultado_retangulos = ut_img.desenhar_retangulos('./imagens/{}'.format(imagem_escolhida), contagem_objetos)
        
        time.sleep(1)
        
        # Realiza a plotagem das imagens
        progress.update(task, advance=1, description='[cyan]Plotando as imagens...')
        ut_img.plotagem_imagem(Imagem_Original, imagem_resultado, imagem_resultado_retangulos)
        
    end_time = time.time() - start_time - 3
        
    # Salva a imagem na pasta de resultados
    if args.save:
        pass
    
    if args.info:
        ut_code.print_infos(end_time, tipo, imagem_escolhida, len(contagem_objetos))
        
    if args.url:
        ut_img.deletar_imagem(imagem_escolhida)
    
    
if __name__ == '__main__':
    
    # Verifica se os argumentos foram passados corretamente
    args = parser.parse_args()
    
    # Verifica se as bibliotecas necessárias estão instaladas
    lib_checker.check_library()
    
    ut_code.clear_terminal()
    ut_code.print_title()
    
    if args.url:
        ut_img.download_imagem(args)
    
    # Inicializa a console
    console = Console()
    
    # Lista as imagens disponíveis na pasta
    imagens_disponiveis = ut_img.lista_imagens_pasta('./imagens', console)
    
    # Escolhe uma imagem para aplicar o método de Otsu
    imagem_escolhida = ut_img.escolher_imagens(imagens_disponiveis, console)
        
    contador_de_objetos(imagem_escolhida, 'Contagem de Objetos')