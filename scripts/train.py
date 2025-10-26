"""
Script principal para executar o pipeline completo do HuMob Challenge.

🆕 NOVO: FINE-TUNING + MLFLOW IMPLEMENTADOS!

CORREÇÕES IMPLEMENTADAS baseadas na análise:
1. ✅ Sequências temporais adequadas (sequence_length > 1) para LSTM fazer sentido
2. ✅ Uso correto da classe ExternalInformationFusionNormalized (dados já normalizados)
3. ✅ Dataset que trabalha com dados já normalizados
4. ✅ Rollout para múltiplos passos (15 dias × 48 slots para HuMob)
5. ✅ Discretização final para grid [0,199]
6. ✅ Cluster centers calculados corretamente
7. ✅ Pipeline completo com treino, validação e submissão
8. 🆕 FINE-TUNING sequencial A → B → C → D
9. 🔬 MLFLOW para tracking de experimentos acadêmicos

PREMISSAS:
- Dados já normalizados conforme especificado:
  * x_norm, y_norm: [0,1] (MinMaxScaler)
  * d_norm: [0,1] (normalização linear) 
  * t_sin, t_cos: [-1,1] (codificação circular)
  * POI_norm: [0,1] (log1p + normalização por categoria)
  * city_encoded: {0,1,2,3} (label encoding)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import os
warnings.filterwarnings('ignore')

# Importa módulos locais (assumindo que estão no mesmo diretório)
try:
    from src.models.humob_model import HuMobModel, discretize_coordinates
    from src.data.dataset import create_humob_loaders, create_test_loader
    from src.training.train import compute_cluster_centers, train_humob_model, evaluate_model
    from src.training.pipeline import run_full_pipeline, generate_humob_submission
    from src.training.finetune import finetune_model, sequential_finetuning, compare_models_performance  # 🆕 NOVO!
    from src.utils.mlflow_tracker import setup_mlflow_for_humob, get_experiment_summary_for_paper  # 🔬 MLflow
    from src.utils.pytorch_compat import load_checkpoint_safe, check_pytorch_version_compatibility  # 🔧 Compatibility
except ImportError as e:
    print(f"❌ Erro importando módulos: {e}")
    print("📋 Certifique-se de que os arquivos estão no mesmo diretório:")
    print("   - external_information.py")
    print("   - partial_information.py") 
    print("   - humob_model.py")
    print("   - humob_dataset.py")
    print("   - humob_training.py (CORRIGIDO)")
    print("   - humob_pipeline.py (CORRIGIDO)")
    print("   - humob_finetuning.py (NOVO)")
    print("   - mlflow_utils.py (NOVO - MLflow)")
    print("   - pytorch_compatibility.py (NOVO - Compatibility)")
    exit(1)


def quick_test(parquet_path: str, device: torch.device):
    """Teste rápido para verificar se o pipeline está funcionando."""
    print("🧪 TESTE RÁPIDO DO PIPELINE")
    print("=" * 40)
    
    try:
        # 1. Testa cluster centers
        print("1. Testando cálculo de cluster centers...")
        centers = compute_cluster_centers(
            parquet_path=parquet_path,
            cities=["A"],
            n_clusters=32,  # Pequeno para teste
            sample_size=10000,
            save_path="test_centers.npy"
        )
        print(f"   ✅ Centers: {centers.shape}")
        
        # 2. Testa criação de dataset
        print("2. Testando dataset...")
        train_loader, val_loader = create_humob_loaders(
            parquet_path=parquet_path,
            cities=["A"],
            batch_size=16,
            sequence_length=6  # Pequeno para teste
        )
        
        # Tenta uma amostra
        for batch in train_loader:
            uid, d_norm, t_sin, t_cos, city, poi_norm, coords_seq, target_coords = batch
            print(f"   ✅ Batch shape: {coords_seq.shape}")
            print(f"   ✅ Ranges: d_norm=[{d_norm.min():.3f},{d_norm.max():.3f}], "
                  f"coords=[{coords_seq.min():.3f},{coords_seq.max():.3f}]")
            break
        
        # 3. Testa modelo
        print("3. Testando modelo...")
        model = HuMobModel(
            n_users=10000,  # Pequeno para teste
            n_cities=4,
            cluster_centers=centers.to(device),
            sequence_length=6
        ).to(device)
        
        # Forward test
        with torch.no_grad():
            uid_test = torch.randint(0, 1000, (2,), device=device)
            d_norm_test = torch.rand(2, device=device)
            t_sin_test = torch.randn(2, device=device)
            t_cos_test = torch.randn(2, device=device)
            city_test = torch.randint(0, 4, (2,), device=device)
            poi_test = torch.rand(2, 85, device=device)
            coords_seq_test = torch.rand(2, 6, 2, device=device)
            
            pred = model.forward_single_step(
                uid_test, d_norm_test, t_sin_test, t_cos_test,
                city_test, poi_test, coords_seq_test
            )
            print(f"   ✅ Predição shape: {pred.shape}")
            print(f"   ✅ Predição range: [{pred.min():.3f}, {pred.max():.3f}]")
        
        print("\n🎉 TESTE RÁPIDO PASSOU! Pipeline está funcionando.")
        return True
        
    except Exception as e:
        print(f"\n❌ TESTE RÁPIDO FALHOU: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_minimal_example(parquet_path: str, use_mlflow: bool = False):
    """Executa um exemplo mínimo para demonstração."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🚀 EXEMPLO MÍNIMO HUMOB")
    if use_mlflow:
        print("🔬 COM TRACKING MLFLOW")
    print(f"Device: {device}")
    print("=" * 40)
    
    # 🔬 MLflow - Setup se solicitado
    mlflow_tracker = None
    if use_mlflow:
        try:
            mlflow_tracker = setup_mlflow_for_humob("HuMob_Minimal_Test")
            print("✅ MLflow configurado!")
        except Exception as e:
            print(f"⚠️ Erro MLflow: {e}")
    
    # Teste rápido primeiro
    if not quick_test(parquet_path, device):
        print("❌ Teste rápido falhou. Verifique os dados e imports.")
        return None
    
    print("\n🏃 Executando pipeline mínimo...")
    
    # Pipeline com configurações mínimas
    results = run_full_pipeline(
        parquet_path=parquet_path,
        device=device,
        n_clusters=128,     # Reduzido
        n_epochs=2,        # Poucas épocas 
        sequence_length=8,  # Histórico pequeno
        batch_size=32,
        learning_rate=2e-3,
        n_users_A=100_000
    )
    
    if results and results.get('checkpoint_path'):
        print("\n📄 Gerando submissão de exemplo...")
        try:
            submission_df = generate_humob_submission(
                parquet_path=parquet_path,
                checkpoint_path=results['checkpoint_path'],
                device=device,
                target_cities=["D"],  # Apenas cidade D para teste
                submission_days=(61, 62),  # Apenas 2 dias para teste
                sequence_length=8,
                output_file="humob_example_submission.csv"
            )
            print(f"✅ Submissão de exemplo gerada: {len(submission_df):,} linhas")
        except Exception as e:
            print(f"⚠️ Erro na submissão: {e}")
    
    return results


def run_full_competition(parquet_path: str, use_mlflow: bool = False):
    """Executa o pipeline completo para competição."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🏆 PIPELINE COMPLETO PARA COMPETIÇÃO")
    if use_mlflow:
        print("🔬 COM TRACKING MLFLOW")
    print(f"Device: {device}")
    print("=" * 50)
    
    # 🔬 MLflow - Setup se solicitado
    mlflow_tracker = None
    if use_mlflow:
        try:
            mlflow_tracker = setup_mlflow_for_humob("HuMob_Competition")
            print("✅ MLflow configurado!")
        except Exception as e:
            print(f"⚠️ Erro MLflow: {e}")
    
    # Configurações otimizadas para competição
    results = run_full_pipeline(
        parquet_path=parquet_path,
        device=device,
        n_clusters=512,      # Mais centros para melhor precisão
        n_epochs=8,         # Mais épocas
        sequence_length=24,  # Mais histórico (12 horas)
        batch_size=64,
        learning_rate=1e-3,
        n_users_A=100_000
    )
    
    if results and results.get('checkpoint_path'):
        print("\n📄 Gerando submissão final...")
        submission_df = generate_humob_submission(
            parquet_path=parquet_path,
            checkpoint_path=results['checkpoint_path'],
            device=device,
            target_cities=["B", "C", "D"],
            submission_days=(61, 75),  # 15 dias completos
            sequence_length=24,
            output_file="humob_final_submission.csv"
        )
        print(f"🎯 Submissão final: {len(submission_df):,} predições")
        print("📧 Pronto para envio ao HuMob Challenge!")
    
    return results


def run_finetuning_example(parquet_path: str, use_mlflow: bool = False):
    """🆕 NOVO: Executa fine-tuning sequencial com opção de MLflow."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎯 FINE-TUNING SEQUENCIAL HUMOB")
    if use_mlflow:
        print("🔬 COM TRACKING MLFLOW")
    print(f"Device: {device}")
    print("=" * 50)
    
    # 🔬 MLflow - Configura tracker se solicitado
    mlflow_tracker = None
    if use_mlflow:
        try:
            mlflow_tracker = setup_mlflow_for_humob("HuMob_Challenge_Paper")
            print("✅ MLflow configurado com sucesso!")
        except Exception as e:
            print(f"⚠️ Erro configurando MLflow: {e}")
            print("Continuando sem tracking...")
    
    # Verifica se modelo base existe
    base_model = "humob_model_A.pt"
    if not os.path.exists(base_model):
        print(f"❌ Modelo base não encontrado: {base_model}")
        print("Execute primeiro a opção 2 ou 3 para treinar o modelo base em A")
        return None
    
    print(f"📂 Modelo base encontrado: {base_model}")
    
    # Fine-tuning sequencial com configurações moderadas
    results = sequential_finetuning(
        parquet_path=parquet_path,
        base_checkpoint=base_model,
        cities=["B", "C", "D"],
        device=device,
        n_epochs_per_city=3,
        learning_rate=5e-5,
        sequence_length=24,
        # 🔬 MLflow - Passa tracker
        mlflow_tracker=mlflow_tracker
    )
    
    # Comparação de performance
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    if successful > 0:
        print(f"\n🔍 COMPARANDO PERFORMANCE...")
        
        # Monta lista de checkpoints para comparar
        checkpoints = {'Zero-shot (A apenas)': base_model}
        
        for city, result in results.items():
            if result['status'] == 'success':
                checkpoints[f'Fine-tuned {city}'] = result['checkpoint']
        
        comparison = compare_models_performance(
            parquet_path=parquet_path,
            checkpoints=checkpoints,
            device=device,
            n_samples=2000,
            # 🔬 MLflow - Passa tracker
            mlflow_tracker=mlflow_tracker
        )
        
        # Se o fine-tuning foi bem-sucedido, oferece gerar submissão
        if successful == len(["B", "C", "D"]):
            print(f"\n🎉 FINE-TUNING COMPLETO!")
            
            if use_mlflow:
                print("🔬 Dados salvos no MLflow! Para visualizar:")
                print("   mlflow ui --backend-store-uri ./mlruns")
                print("   Acesse: http://localhost:5000")
            
            response = input("\n📄 Gerar submissão com modelos fine-tuned? (y/n): ").strip().lower()
            if response == 'y':
                # Gera submissão usando modelo da última cidade (D)
                final_checkpoint = results["D"]["checkpoint"]
                submission_df = generate_humob_submission(
                    parquet_path=parquet_path,
                    checkpoint_path=final_checkpoint,
                    device=device,
                    target_cities=["B", "C", "D"],
                    submission_days=(61, 75),
                    sequence_length=24,
                    output_file="humob_finetuned_submission.csv"
                )
                print(f"✅ Submissão fine-tuned gerada: {len(submission_df):,} predições")
    
    return results


def run_single_city_finetuning(parquet_path: str, use_mlflow: bool = False):
    """🆕 NOVO: Fine-tuning em uma cidade específica."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎯 FINE-TUNING CIDADE ESPECÍFICA")
    if use_mlflow:
        print("🔬 COM TRACKING MLFLOW")
    print("=" * 40)
    
    # 🔬 MLflow - Setup se solicitado
    mlflow_tracker = None
    if use_mlflow:
        try:
            mlflow_tracker = setup_mlflow_for_humob("HuMob_Single_City")
            print("✅ MLflow configurado!")
        except Exception as e:
            print(f"⚠️ Erro MLflow: {e}")
    
    # Verifica modelo base
    base_model = "humob_model_A.pt"
    if not os.path.exists(base_model):
        print(f"❌ Modelo base não encontrado: {base_model}")
        print("Execute primeiro a opção 2 ou 3 para treinar o modelo base em A")
        return None
    
    # Seleciona cidade
    print("Cidades disponíveis para fine-tuning: B, C, D")
    city = input("Digite a cidade (B/C/D): ").strip().upper()
    
    if city not in ["B", "C", "D"]:
        print("❌ Cidade inválida")
        return None
    
    print(f"🎯 Iniciando fine-tuning na cidade {city}...")
    
    # Fine-tuning
    try:
        model, train_losses, val_losses = finetune_model(
            parquet_path=parquet_path,
            pretrained_checkpoint=base_model,
            target_city=city,
            device=device,
            n_epochs=3,
            learning_rate=5e-5,
            sequence_length=24,
            # 🔬 MLflow - Passa tracker
            mlflow_tracker=mlflow_tracker
        )
        
        print(f"\n✅ Fine-tuning cidade {city} concluído!")
        
        if use_mlflow:
            print("🔬 Dados salvos no MLflow!")
        
        # Compara com zero-shot
        checkpoints = {
            'Zero-shot (A apenas)': base_model,
            f'Fine-tuned {city}': f'humob_model_finetuned_{city}.pt'
        }
        
        comparison = compare_models_performance(
            parquet_path=parquet_path,
            checkpoints=checkpoints,
            test_cities=[city],
            device=device,
            n_samples=3000,
            # 🔬 MLflow - Passa tracker
            mlflow_tracker=mlflow_tracker
        )
        
        return {
            'city': city,
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'checkpoint': f'humob_model_finetuned_{city}.pt',
            'comparison': comparison
        }
        
    except Exception as e:
        print(f"❌ Erro durante fine-tuning: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_mlflow_complete_experiment(parquet_path: str):
    """🔬 Execução completa com tracking MLflow para paper."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔬 EXPERIMENTO COMPLETO PARA PAPER")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Setup MLflow
    try:
        mlflow_tracker = setup_mlflow_for_humob("HuMob_Challenge_Paper")
        print("✅ MLflow configurado para paper!")
        print("📊 Dados serão organizados para publicação acadêmica")
    except Exception as e:
        print(f"❌ Erro configurando MLflow: {e}")
        print("💡 Instale MLflow: pip install mlflow")
        return None
    
    print(f"\n📋 PLANO DE EXPERIMENTOS:")
    print(f"   1. Treinamento base na cidade A (8 épocas)")
    print(f"   2. Fine-tuning sequencial B→C→D (3 épocas cada)")
    print(f"   3. Avaliação e comparação automática")
    print(f"   4. Geração de plots e tabelas para paper")
    print(f"   5. Exportação de dados organizados")
    
    confirm = input(f"\n🚀 Iniciar experimento completo? (Pode demorar 6-12h) (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Experimento cancelado")
        return None
    
    # ETAPA 1: Treinamento base com tracking completo
    print(f"\n🏋️ ETAPA 1/5: TREINAMENTO BASE (CIDADE A)")
    print("=" * 50)
    
    # Calcula cluster centers
    centers = compute_cluster_centers(
        parquet_path=parquet_path,
        cities=["A"],
        n_clusters=32,
        save_path="centers_A_paper.npy"
    )
    
    # Treinamento com tracking MLflow
    model, train_losses, val_losses = train_humob_model(
        parquet_path=parquet_path,
        cluster_centers=centers,
        device=device,
        cities=["A"],
        n_epochs = 1 #8,  # Mais épocas para melhor resultado
        learning_rate=1e-3,
        batch_size=16#32,
        sequence_length=6#24,
        n_users=100_000,
        save_path="humob_model_A_paper.pt",
        # 🔬 MLflow tracking
        mlflow_tracker=mlflow_tracker
    )
    
    print("✅ Treinamento base concluído!")
    
    # ETAPA 2: Fine-tuning sequencial 
    print(f"\n🎯 ETAPA 2/5: FINE-TUNING SEQUENCIAL")
    print("=" * 50)
    
    results = sequential_finetuning(
        parquet_path=parquet_path,
        base_checkpoint="humob_model_A_paper.pt",
        cities=["B", "C", "D"],
        device=device,
        n_epochs_per_city=3,
        learning_rate=5e-5,  # LR baixo para fine-tuning
        sequence_length=24,
        mlflow_tracker=mlflow_tracker
    )
    
    successful = sum(1 for r in results.values() if r['status'] == 'success')
    print(f"✅ Fine-tuning: {successful}/3 cidades concluídas")
    
    # ETAPA 3: Avaliação e comparação
    print(f"\n📊 ETAPA 3/5: AVALIAÇÃO E COMPARAÇÃO")
    print("=" * 50)
    
    if successful > 0:
        # Monta checkpoints para comparar
        checkpoints = {'Zero-shot (A apenas)': "humob_model_A_paper.pt"}
        for city, result in results.items():
            if result['status'] == 'success':
                checkpoints[f'Fine-tuned {city}'] = result['checkpoint']
        
        comparison = compare_models_performance(
            parquet_path=parquet_path,
            checkpoints=checkpoints,
            device=device,
            n_samples=5000,  # Mais amostras para avaliação final
            mlflow_tracker=mlflow_tracker
        )
        print("✅ Comparação concluída!")
    
    # ETAPA 4: Geração de plots e dados
    print(f"\n📈 ETAPA 4/5: GERANDO DADOS PARA PAPER")
    print("=" * 50)
    
    try:
        from export_paper_data import PaperDataExporter
        
        exporter = PaperDataExporter()
        paper_data = exporter.export_all_data()
        
        if paper_data:
            print("✅ Dados para paper exportados!")
            print(f"📁 Arquivos salvos em: {paper_data['output_dir']}")
        else:
            print("⚠️ Erro exportando dados para paper")
    
    except ImportError:
        print("⚠️ export_paper_data.py não encontrado")
        print("💡 Use mlflow ui para visualizar resultados")
    
    # ETAPA 5: Submissão final
    print(f"\n📄 ETAPA 5/5: SUBMISSÃO FINAL")
    print("=" * 50)
    
    if successful == len(["B", "C", "D"]):
        response = input("Gerar submissão final com melhor modelo? (y/n): ").strip().lower()
        if response == 'y':
            final_checkpoint = results["D"]["checkpoint"]
            submission_df = generate_humob_submission(
                parquet_path=parquet_path,
                checkpoint_path=final_checkpoint,
                device=device,
                target_cities=["B", "C", "D"],
                submission_days=(61, 75),
                sequence_length=24,
                output_file="humob_paper_submission.csv"
            )
            print(f"✅ Submissão final: {len(submission_df):,} predições")
    
    # Relatório final
    print(f"\n🎉 EXPERIMENTO COMPLETO FINALIZADO!")
    print("=" * 50)
    print("🔬 DADOS DISPONÍVEIS NO MLFLOW:")
    print("   - Hiperparâmetros de todos os experimentos")
    print("   - Métricas de treinamento (loss, fusion weights)")
    print("   - Comparações automáticas zero-shot vs fine-tuned")
    print("   - Gráficos publication-ready")
    print("   - Tabelas LaTeX formatadas")
    
    print(f"\n📊 PARA VISUALIZAR:")
    print(f"   mlflow ui --backend-store-uri ./mlruns")
    print(f"   Acesse: http://localhost:5000")
    
    print(f"\n📋 ARQUIVOS GERADOS:")
    for pattern in ["*.pt", "*.csv", "paper_data/*"]:
        files = list(Path(".").glob(pattern))
        for f in files[:5]:  # Mostra até 5 arquivos
            print(f"   - {f}")
        if len(files) > 5:
            print(f"   ... e mais {len(files)-5} arquivos")
    
    print(f"\n🎯 PRÓXIMOS PASSOS PARA PAPER:")
    print(f"   1. Revisar dados em paper_data/")
    print(f"   2. Incluir figuras (.pdf) no paper")  
    print(f"   3. Usar tabelas LaTeX (.tex)")
    print(f"   4. Citar estatísticas do summary_statistics.json")
    
    return {
        'mlflow_tracker': mlflow_tracker,
        'results': results,
        'comparison': comparison if 'comparison' in locals() else None
    }


def main():
    """Função principal com menu interativo COMPLETO com MLflow."""
    
    # Verifica compatibilidade PyTorch
    print("🔧 VERIFICANDO COMPATIBILIDADE...")
    try:
        compat_info = check_pytorch_version_compatibility(verbose=False)
        if compat_info['needs_safe_loading']:
            print("⚠️  PyTorch 2.6+ detectado - usando carregamento seguro")
    except:
        pass
    
    # Configuração do arquivo (AJUSTE AQUI)
    parquet_file = "data/humob_all_cities_v2_normalized.parquet"
    
    if not os.path.exists(parquet_file):
        print(f"❌ Arquivo não encontrado: {parquet_file}")
        print("📋 Certifique-se de:")
        print("   1. Ter executado a normalização dos dados")
        print("   2. O arquivo ter as colunas corretas:")
        print("      - uid, city_encoded, d_norm, t_sin, t_cos")
        print("      - x_norm, y_norm, POI_norm")
        return
    
    print(f"📁 Arquivo encontrado: {parquet_file}")
    print()
    
    # Menu COMPLETO
    print("Escolha uma opção:")
    print("1. 🧪 Teste rápido (verifica se tudo está funcionando)")
    print("2. 🏃 Exemplo mínimo (pipeline pequeno para demonstração)")  
    print("3. 🏆 Pipeline completo (para submissão final)")
    print("4. 🎯 Fine-tuning sequencial B→C→D (sem MLflow)")
    print("5. 🎪 Fine-tuning cidade específica (sem MLflow)")
    print("6. 🔍 Avaliar modelos existentes (sem treinar)")
    print("7. 🔬 Fine-tuning COM MLflow (recomendado para paper)")  
    print("8. 🎓 Experimento COMPLETO para paper (6-12h)")     
    print("9. 📊 Ver resumo dos experimentos MLflow")              
    print("10. 📈 Exportar dados para paper (tabelas/gráficos)")
    
    try:
        choice = input("\nDigite sua escolha (1-10): ").strip()
        
        if choice == "1":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            quick_test(parquet_file, device)
            
        elif choice == "2":
            print("\n🔬 Usar MLflow? (recomendado)")
            use_mlflow = input("Digite 'y' para usar MLflow: ").strip().lower() == 'y'
            run_minimal_example(parquet_file, use_mlflow=use_mlflow)
            
        elif choice == "3":
            print("\n🔬 Usar MLflow? (recomendado)")
            use_mlflow = input("Digite 'y' para usar MLflow: ").strip().lower() == 'y'
            run_full_competition(parquet_file, use_mlflow=use_mlflow)
            
        elif choice == "4":
            run_finetuning_example(parquet_file, use_mlflow=False)
            
        elif choice == "5":
            print("\n🔬 Usar MLflow?")
            use_mlflow = input("Digite 'y' para usar MLflow: ").strip().lower() == 'y'
            run_single_city_finetuning(parquet_file, use_mlflow=use_mlflow)
            
        elif choice == "6":
            # Avalia modelos existentes
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Busca modelos disponíveis
            checkpoints = {}
            
            if os.path.exists("humob_model_A.pt"):
                checkpoints['Zero-shot (A apenas)'] = "humob_model_A.pt"
            
            for city in ["B", "C", "D"]:
                checkpoint = f"humob_model_finetuned_{city}.pt"
                if os.path.exists(checkpoint):
                    checkpoints[f'Fine-tuned {city}'] = checkpoint
            
            if not checkpoints:
                print("❌ Nenhum modelo encontrado. Execute treino primeiro.")
            else:
                print(f"📊 Encontrados {len(checkpoints)} modelos:")
                for name in checkpoints:
                    print(f"   - {name}")
                
                print("\n🔬 Usar MLflow?")
                use_mlflow = input("Digite 'y' para usar MLflow: ").strip().lower() == 'y'
                
                mlflow_tracker = None
                if use_mlflow:
                    try:
                        mlflow_tracker = setup_mlflow_for_humob("HuMob_Evaluation")
                        print("✅ MLflow configurado!")
                    except:
                        pass
                
                comparison = compare_models_performance(
                    parquet_path=parquet_file,
                    checkpoints=checkpoints,
                    device=device,
                    n_samples=3000,
                    mlflow_tracker=mlflow_tracker
                )
        
        elif choice == "7":
            print("\n🔬 FINE-TUNING COM MLFLOW PARA PAPER")
            print("Esta opção salva todos os dados no MLflow para análise acadêmica")
            confirm = input("Continuar? (y/n): ").strip().lower()
            if confirm == 'y':
                run_finetuning_example(parquet_file, use_mlflow=True)
        
        elif choice == "8":
            print("\n🎓 EXPERIMENTO COMPLETO PARA PAPER")
            print("Esta opção executa TODOS os experimentos com tracking completo")
            print("⏱️  Tempo estimado: 6-12 horas")
            print("📊 Ideal para gerar dados finais do paper")
            confirm = input("Continuar? (y/n): ").strip().lower()
            if confirm == 'y':
                run_mlflow_complete_experiment(parquet_file)
            
        elif choice == "9":
            print("\n📊 RESUMO DOS EXPERIMENTOS MLFLOW")
            try:
                summary = get_experiment_summary_for_paper()
                if summary:
                    print("\n🔬 Para visualizar detalhes completos:")
                    print("   mlflow ui --backend-store-uri ./mlruns")
                    print("   Acesse: http://localhost:5000")
                else:
                    print("❌ Nenhum experimento MLflow encontrado")
                    print("Execute a opção 7 ou 8 primeiro para gerar dados")
            except Exception as e:
                print(f"❌ Erro acessando MLflow: {e}")
                print("Execute 'pip install mlflow' se não estiver instalado")
        
        # 🆕 NOVA OPÇÃO: Exportar dados para paper
        elif choice == "10":
            print("\n📈 EXPORTAR DADOS PARA PAPER")
            try:
                from export_paper_data import PaperDataExporter
                
                exporter = PaperDataExporter()
                results = exporter.export_all_data()
                
                if results:
                    print("✅ Dados exportados com sucesso!")
                    print("📁 Use os arquivos em paper_data/ para seu artigo")
                else:
                    print("❌ Erro na exportação")
                    print("💡 Execute experimentos MLflow primeiro (opção 7 ou 8)")
                    
            except ImportError:
                print("❌ export_paper_data.py não encontrado")
            except Exception as e:
                print(f"❌ Erro: {e}")
            
        else:
            print("❌ Opção inválida")
            
    except KeyboardInterrupt:
        print("\n🛑 Execução interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()