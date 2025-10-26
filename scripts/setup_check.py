"""
Script para verificar se todos os arquivos estão corretos e funcionando.
Execute este script antes de rodar o pipeline principal.
"""

import os
import sys
import traceback

def check_file_exists(filename):
    """Verifica se um arquivo existe."""
    if os.path.exists(filename):
        print(f"✅ {filename} - encontrado")
        return True
    else:
        print(f"❌ {filename} - NÃO ENCONTRADO")
        return False

def test_imports():
    """Testa se todos os imports funcionam."""
    print("\n🔍 Testando imports...")
    
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        import pandas as pd
        print("✅ Dependências básicas (torch, numpy, pandas) - OK")
    except ImportError as e:
        print(f"❌ Erro nas dependências básicas: {e}")
        return False
    
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        print("✅ PyArrow - OK")
    except ImportError:
        print("⚠️  PyArrow não encontrado. Instale com: pip install pyarrow")
        return False
    
    try:
        from sklearn.cluster import KMeans
        print("✅ Scikit-learn - OK")
    except ImportError:
        print("⚠️  Scikit-learn não encontrado. Instale com: pip install scikit-learn")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib - OK")
    except ImportError:
        print("⚠️  Matplotlib não encontrado. Instale com: pip install matplotlib")
        return False
    
    try:
        from tqdm import tqdm
        print("✅ tqdm - OK")
    except ImportError:
        print("⚠️  tqdm não encontrado. Instale com: pip install tqdm")
        return False
    
    return True

def test_custom_modules():
    """Testa se os módulos customizados funcionam."""
    print("\n🧪 Testando módulos customizados...")
    
    try:
        from src.models.external_info import ExternalInformationFusionNormalized, ExternalInformationDense
        print("✅ external_information.py - importado com sucesso")
        
        # Teste básico
        fusion = ExternalInformationFusionNormalized(n_users=100, n_cities=4)
        print(f"✅ ExternalInformationFusionNormalized - out_dim = {fusion.out_dim}")
        
    except Exception as e:
        print(f"❌ Erro em external_information.py: {e}")
        return False
    
    try:
        from src.models.partial_info import CoordLSTM
        print("✅ partial_information.py - importado com sucesso")
        
        # Teste básico
        lstm = CoordLSTM(input_size=2, hidden_size=8)
        print(f"✅ CoordLSTM - output_dim = {lstm.output_dim}")
        
    except Exception as e:
        print(f"❌ Erro em partial_information.py: {e}")
        return False
    
    return True

def test_model_creation():
    """Testa se consegue criar o modelo principal."""
    print("\n🏗️  Testando criação do modelo...")
    
    try:
        # Imports
        import torch
        from src.models.external_info import ExternalInformationFusionNormalized, ExternalInformationDense
        from src.models.partial_info import CoordLSTM
        
        # Dados de teste
        batch_size = 4
        sequence_length = 6
        
        # Cria cluster centers fake
        n_clusters = 16
        centers = torch.rand(n_clusters, 2)  # [0,1]
        
        # Dados de entrada simulados
        uid = torch.randint(0, 100, (batch_size,))
        d_norm = torch.rand(batch_size)
        t_sin = torch.randn(batch_size)
        t_cos = torch.randn(batch_size) 
        city = torch.randint(0, 4, (batch_size,))
        poi_norm = torch.rand(batch_size, 85)
        coords_seq = torch.rand(batch_size, sequence_length, 2)
        
        print(f"   Dados de teste criados: batch_size={batch_size}, seq_len={sequence_length}")
        
        # Testa fusão externa
        fusion = ExternalInformationFusionNormalized(
            n_users=100,
            n_cities=4,
            user_emb_dim=4,
            city_emb_dim=4,
            temporal_dim=4,
            poi_out_dim=4
        )
        
        dense = ExternalInformationDense(in_dim=fusion.out_dim, out_dim=8)
        
        with torch.no_grad():
            static_emb = fusion(uid, d_norm, t_sin, t_cos, city, poi_norm)
            static_red = dense(static_emb)
            print(f"✅ Fusão externa: {static_emb.shape} → {static_red.shape}")
        
        # Testa LSTM
        lstm = CoordLSTM(input_size=2, hidden_size=4, bidirectional=True)
        
        with torch.no_grad():
            dyn_emb = lstm(coords_seq)
            print(f"✅ CoordLSTM: {coords_seq.shape} → {dyn_emb.shape}")
        
        # Verifica compatibilidade de dimensões
        if static_red.shape[1] == dyn_emb.shape[1]:
            print(f"✅ Dimensões compatíveis para fusão: {static_red.shape[1]}")
        else:
            print(f"❌ Dimensões incompatíveis: static={static_red.shape[1]}, dynamic={dyn_emb.shape[1]}")
            return False
            
        print("🎉 Modelo pode ser criado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro criando modelo: {e}")
        traceback.print_exc()
        return False

def show_file_structure():
    """Mostra a estrutura de arquivos necessária."""
    print("\n📁 ESTRUTURA DE ARQUIVOS NECESSÁRIA:")
    print("=" * 50)
    
    required_files = [
        "external_information.py",      # Fusão de informação externa
        "partial_information.py",       # LSTM para coordenadas
        "humob_model.py",              # Modelo principal
        "humob_dataset.py",            # Dataset para dados normalizados
        "humob_training.py",           # Funções de treino
        "humob_pipeline.py",           # Pipeline completo
        "run_humob.py",                # Script principal
        "SEU_ARQUIVO_NORMALIZADO.parquet"  # Seus dados normalizados
    ]
    
    for file in required_files:
        check_file_exists(file)
    
    print("\n📋 COMO OBTER OS ARQUIVOS FALTANTES:")
    print("=" * 50)
    print("1. external_information.py - Fornecido na resposta anterior")
    print("2. partial_information.py - Fornecido na resposta anterior") 
    print("3. humob_model.py - Criado pelo código corrigido")
    print("4. humob_dataset.py - Criado pelo código corrigido")
    print("5. humob_training.py - Criado pelo código corrigido")
    print("6. humob_pipeline.py - Criado pelo código corrigido")
    print("7. run_humob.py - Criado pelo código corrigido")
    print("8. Arquivo de dados - Use seu script de normalização")

def main():
    """Executa todas as verificações."""
    print("🔧 VERIFICAÇÃO DE SETUP - HUMOB CHALLENGE")
    print("=" * 50)
    
    # 1. Verifica arquivos
    print("\n📁 Verificando arquivos...")
    file_check = True
    required_py_files = [
        "external_information.py",
        "partial_information.py"
    ]
    
    for file in required_py_files:
        if not check_file_exists(file):
            file_check = False
    
    # 2. Testa dependências
    deps_ok = test_imports()
    
    # 3. Testa módulos customizados
    modules_ok = test_custom_modules() if file_check else False
    
    # 4. Testa criação de modelo
    model_ok = test_model_creation() if modules_ok else False
    
    # 5. Relatório final
    print("\n📊 RELATÓRIO FINAL")
    print("=" * 30)
    
    if file_check and deps_ok and modules_ok and model_ok:
        print("🎉 SETUP COMPLETO! Tudo funcionando.")
        print("\n✅ Próximos passos:")
        print("   1. Certifique-se de que tem seu arquivo .parquet normalizado")
        print("   2. Ajuste o caminho em run_humob.py")
        print("   3. Execute: python run_humob.py")
        return True
    else:
        print("❌ PROBLEMAS ENCONTRADOS:")
        if not file_check:
            print("   • Arquivos faltando")
        if not deps_ok:
            print("   • Dependências faltando")
        if not modules_ok:
            print("   • Módulos customizados com erro")
        if not model_ok:
            print("   • Erro na criação do modelo")
        
        show_file_structure()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)