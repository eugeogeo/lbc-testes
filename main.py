import os
import glob
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import csv
from datetime import datetime

# ==========================================
# 1. Configurações e Carregamento de Dados
# ==========================================
def carregar_fmd(caminho_base):
    """
    Carrega imagens da base FMD. 
    Espera-se que o diretório tenha subpastas para cada classe (ex: fabric, foliage, glass...).
    """
    imagens = []
    rotulos = []
    classes = os.listdir(caminho_base)
    
    for idx_classe, classe in enumerate(classes):
        caminho_classe = os.path.join(caminho_base, classe)
        if not os.path.isdir(caminho_classe):
            continue
            
        # Busca por múltiplas extensões de imagem para evitar erros
        extensoes = ['*.jpg', '*.JPG', '*.png', '*.jpeg']
        para_cada_imagem = []
        for ext in extensoes:
            para_cada_imagem.extend(glob.glob(os.path.join(caminho_classe, ext)))
            
        for caminho_img in para_cada_imagem:
            img = cv2.imread(caminho_img)
            if img is not None:
                imagens.append(img)
                rotulos.append(idx_classe)
                
    return imagens, np.array(rotulos)

# ==========================================
# 2. Extratores de Características (Parte A)
# ==========================================
def extrair_features_resnet18(imagens):
    """
    Usa a ResNet-18 pré-treinada (ImageNet) como extrator de características.
    A camada Global Average Pooling (GAP) é mantida, removendo apenas o classificador final.
    """
    modelo = models.resnet18(weights='DEFAULT')
    extrator = torch.nn.Sequential(*(list(modelo.children())[:-1]))
    extrator.eval()
    
    transformacao = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features_list = []
    with torch.no_grad():
        for img in imagens:
            img_tensor = transformacao(img).unsqueeze(0)
            features = extrator(img_tensor)
            features_list.append(features.view(-1).numpy())
            
    return np.array(features_list)

def extrair_features_lbp(imagens, raio=3, pontos=24):
    """Extrai características LBP (Local Binary Pattern) uniformes."""
    features_list = []
    for img in imagens:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(img_gray, pontos, raio, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        features_list.append(hist)
    return np.array(features_list)

def extrair_features_glcm(imagens, distancias=[1, 3], angulos=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Extrai características GLCM (Gray-Level Co-Occurrence Matrix)."""
    features_list = []
    for img in imagens:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray_reduzida = (img_gray / 8).astype(np.uint8) 
        
        glcm = graycomatrix(img_gray_reduzida, distances=distancias, angles=angulos, levels=32, symmetric=True, normed=True)
        
        contraste = graycoprops(glcm, 'contrast').flatten()
        dissimilaridade = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneidade = graycoprops(glcm, 'homogeneity').flatten()
        energia = graycoprops(glcm, 'energy').flatten()
        correlacao = graycoprops(glcm, 'correlation').flatten()
        
        features = np.hstack([contraste, dissimilaridade, homogeneidade, energia, correlacao])
        features_list.append(features)
    return np.array(features_list)

# ==========================================
# 3. Novas Arquiteturas (Parte B)
# ==========================================
class RedeRasaTextura(nn.Module):
    def __init__(self):
        super(RedeRasaTextura, self).__init__()
        # 3 camadas convolucionais seguidas de MaxPool
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Global Average Pooling (GAP)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.gap(x)
        x = x.view(x.size(0), -1) 
        return x

def extrair_features_rede_rasa(imagens):
    """Extrai características usando a Rede Rasa com inicialização aleatória."""
    modelo = RedeRasaTextura()
    modelo.eval()
    
    transformacao = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features_list = []
    with torch.no_grad():
        for img in imagens:
            img_tensor = transformacao(img).unsqueeze(0)
            features = modelo(img_tensor)
            features_list.append(features.view(-1).numpy())
            
    return np.array(features_list)

def extrair_features_resnet18_nao_treinada(imagens):
    """Usa a ResNet-18 SEM pré-treinamento (pesos aleatórios)."""
    modelo = models.resnet18(weights=None) 
    extrator = torch.nn.Sequential(*(list(modelo.children())[:-1]))
    extrator.eval()
    
    transformacao = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    features_list = []
    with torch.no_grad():
        for img in imagens:
            img_tensor = transformacao(img).unsqueeze(0)
            features = extrator(img_tensor)
            features_list.append(features.view(-1).numpy())
            
    return np.array(features_list)

# ==========================================
# 4. Avaliação e Salvamento (CSV)
# ==========================================
def salvar_resultado_csv(nome_experimento, media, desvio, nome_arquivo="resultados_experimentos.csv"):
    arquivo_existe = os.path.isfile(nome_arquivo)
    
    with open(nome_arquivo, mode='a', newline='', encoding='utf-8') as arquivo:
        writer = csv.writer(arquivo)
        if not arquivo_existe:
            writer.writerow(["Data/Hora", "Experimento", "Acurácia Média (%)", "Desvio Padrão (%)"])
            
        data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([data_hora, nome_experimento, f"{media:.2f}", f"{desvio:.2f}"])

def avaliar_modelo(features, rotulos, nome_experimento):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    pipeline_svm = make_pipeline(StandardScaler(), SVC(kernel='linear', class_weight='balanced'))
    
    acuracias = []
    
    for train_index, test_index in skf.split(features, rotulos):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = rotulos[train_index], rotulos[test_index]
        
        pipeline_svm.fit(X_train, y_train)
        y_pred = pipeline_svm.predict(X_test)
        
        acuracias.append(accuracy_score(y_test, y_pred))
        
    media_acc = np.mean(acuracias) * 100
    desvio_acc = np.std(acuracias) * 100
    
    print(f"--- Resultados: {nome_experimento} ---")
    print(f"Acurácia Média: {media_acc:.2f}%")
    print(f"Desvio Padrão:  {desvio_acc:.2f}%\n")
    
    salvar_resultado_csv(nome_experimento, media_acc, desvio_acc)

# ==========================================
# 5. Execução Principal com Cache
# ==========================================
def obter_features(nome_feature, funcao_extracao, imagens, arquivo_cache):
    """Verifica se as características já foram salvas. Se sim, carrega. Se não, extrai e salva."""
    if os.path.exists(arquivo_cache):
        print(f"[{nome_feature}] Carregando do cache ({arquivo_cache})...")
        return np.load(arquivo_cache)
    else:
        print(f"[{nome_feature}] Extraindo características... (Isso pode demorar!)")
        features = funcao_extracao(imagens)
        np.save(arquivo_cache, features)
        print(f"[{nome_feature}] Salvo no cache.")
        return features

if __name__ == "__main__":
    caminho_base_fmd = "./FMD_dataset/image"
    pasta_cache = "./cache_features"
    
    os.makedirs(pasta_cache, exist_ok=True)
    
    cache_rotulos = os.path.join(pasta_cache, "rotulos.npy")
    cache_resnet = os.path.join(pasta_cache, "features_resnet.npy")
    cache_lbp = os.path.join(pasta_cache, "features_lbp.npy")
    cache_glcm = os.path.join(pasta_cache, "features_glcm.npy")
    cache_rede_rasa = os.path.join(pasta_cache, "features_rede_rasa.npy")
    cache_resnet_vazia = os.path.join(pasta_cache, "features_resnet_vazia.npy")
    
    if os.path.exists(cache_rotulos):
        print("Carregando rótulos do cache...")
        rotulos = np.load(cache_rotulos)
        # Só precisamos das imagens se algum cache estiver faltando
        if not (os.path.exists(cache_resnet) and os.path.exists(cache_lbp) and 
                os.path.exists(cache_glcm) and os.path.exists(cache_rede_rasa) and 
                os.path.exists(cache_resnet_vazia)):
            print("Carregando imagens do disco pois há caches faltando...")
            imagens, _ = carregar_fmd(caminho_base_fmd)
        else:
            imagens = [] 
    else:
        print("Carregando imagens do disco...")
        imagens, rotulos = carregar_fmd(caminho_base_fmd)
        np.save(cache_rotulos, rotulos)

    # PARTE A - MÉTODOS DE REFERÊNCIA (Baselines)
    print("\n=== PARTE A: Baselines ===")
    features_resnet = obter_features("ResNet-18", extrair_features_resnet18, imagens, cache_resnet)
    avaliar_modelo(features_resnet, rotulos, "ResNet-18 (Pré-treinada) + SVM")
    
    features_lbp = obter_features("LBP", extrair_features_lbp, imagens, cache_lbp)
    avaliar_modelo(features_lbp, rotulos, "LBP + SVM")
    
    features_glcm = obter_features("GLCM", extrair_features_glcm, imagens, cache_glcm)
    avaliar_modelo(features_glcm, rotulos, "GLCM + SVM")

    # PARTE B - REDES NÃO TREINADAS
    print("\n=== PARTE B: Arquiteturas Sem Pesos (Aleatórias) ===")
    features_rasa = obter_features("Rede Rasa (Aleatória)", extrair_features_rede_rasa, imagens, cache_rede_rasa)
    avaliar_modelo(features_rasa, rotulos, "Rede Rasa 3 Camadas + SVM")

    features_resnet_vazia = obter_features("ResNet-18 (Vazia)", extrair_features_resnet18_nao_treinada, imagens, cache_resnet_vazia)
    avaliar_modelo(features_resnet_vazia, rotulos, "ResNet-18 (Sem Pesos ImageNet) + SVM")