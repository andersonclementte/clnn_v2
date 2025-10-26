import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Importa funções necessárias
from src.training.train import evaluate_model
from src.training.pipeline import generate_humob_submission
# 🔬 MLflow imports
from src.utils.mlflow_tracker import setup_mlflow_for_humob, get_experiment_summary_for_paper


def evaluate_all_models_with_mlflow(
    parquet_path: str,
    target_cities: list[str] = ["B", "C", "D"],
    device: torch.device = None,
    use_mlflow: bool = True
):
    """
    Avalia TODOS os modelos disponíveis com tracking MLflow opcional.
    
    Analogia: Como um "teste de laboratório" que compara todos os
    tratamentos (modelos) ao mesmo tempo e documenta os resultados.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🔍 AVALIAÇÃO COMPLETA DE MODELOS")
    if use_mlflow:
        print("🔬 COM TRACKING MLFLOW")
    print("=" * 50)
    
    # 🔬 MLflow - Setup do tracker
    mlflow_tracker = None
    if use_mlflow:
        try:
            mlflow_tracker = setup_mlflow_for_humob("HuMob_Evaluation")
            print("✅ MLflow configurado para avaliação!")
        except Exception as e:
            print(f"⚠️ Erro configurando MLflow: {e}")
            print("Continuando sem tracking...")
    
    # Busca TODOS os modelos disponíveis automaticamente
    print("🔎 Buscando modelos disponíveis...")
    
    checkpoints = {}
    
    # Modelo base (zero-shot)
    if os.path.exists("humob_model_A.pt"):
        checkpoints['Zero-shot (A apenas)'] = "humob_model_A.pt"
        print("   ✅ Encontrado: Zero-shot (A apenas)")
    
    # Modelos fine-tuned
    for city in ["B", "C", "D"]:
        checkpoint_path = f"humob_model_finetuned_{city}.pt"
        if os.path.exists(checkpoint_path):
            checkpoints[f'Fine-tuned {city}'] = checkpoint_path
            print(f"   ✅ Encontrado: Fine-tuned {city}")
    
    if not checkpoints:
        print("❌ Nenhum modelo encontrado!")
        print("Execute o treinamento primeiro com run_humob.py")
        return None
    
    print(f"\n📊 Total de {len(checkpoints)} modelos para avaliar")
    
    # Avalia cada modelo em cada cidade
    results = {}
    all_results = {}
    
    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n🔍 Avaliando {model_name}...")
        
        model_type = "fine_tuned" if "fine" in model_name.lower() else "zero_shot"
        results[model_name] = {}
        
        for city in target_cities:
            try:
                print(f"   Testando em cidade {city}...")
                mse, cell_error = evaluate_model(
                    parquet_path=parquet_path,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    cities=[city],
                    n_samples=5000,  # Amostra maior para avaliação final
                    sequence_length=24,
                    mlflow_tracker=mlflow_tracker if use_mlflow else None,
                    model_type=model_type
                )
                
                results[model_name][city] = {
                    'mse': mse, 
                    'cell_error': cell_error,
                    'distance_km': cell_error * 0.5  # Conversão para km
                }
                
                print(f"   ✅ {city}: MSE={mse:.4f}, Erro={cell_error:.2f} células ({cell_error*0.5:.2f}km)")
                
            except Exception as e:
                print(f"   ❌ {city}: Erro - {e}")
                results[model_name][city] = {
                    'mse': float('inf'), 
                    'cell_error': float('inf'),
                    'distance_km': float('inf')
                }
    
    # 🔬 MLflow - Log comparação completa
    if mlflow_tracker and use_mlflow:
        comp_run_id = mlflow_tracker.start_comparison_run(checkpoints)
        mlflow_tracker.log_model_comparison(results)
        mlflow_tracker.create_results_comparison_plot(results, "cell_error")
    
    # Gera relatório final com ranking
    print(f"\n🏆 RANKING GERAL")
    print("=" * 40)
    
    # Calcula performance média de cada modelo
    model_averages = []
    
    for model_name, cities_data in results.items():
        valid_errors = [
            v['cell_error'] for v in cities_data.values() 
            if v['cell_error'] != float('inf')
        ]
        
        if valid_errors:
            avg_error = np.mean(valid_errors)
            avg_km = avg_error * 0.5
            model_averages.append((model_name, avg_error, avg_km, len(valid_errors)))
    
    # Ordena por erro médio (menor é melhor)
    model_averages.sort(key=lambda x: x[1])
    
    # Mostra ranking
    for i, (model_name, avg_error, avg_km, n_cities) in enumerate(model_averages):
        ranking = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}º"
        print(f"{ranking} {model_name:20s}: MSE={avg_error:.4f}, Células={avg_error:.2f}, "
              f"Distância={avg_km:.2f}km ({n_cities} cidades)")
    
    # Análise de melhoria do fine-tuning
    zero_shot_models = [x for x in model_averages if "zero" in x[0].lower()]
    ft_models = [x for x in model_averages if "fine" in x[0].lower()]
    
    if zero_shot_models and ft_models:
        best_zero_shot = min(zero_shot_models, key=lambda x: x[1])
        best_ft = min(ft_models, key=lambda x: x[1])
        
        improvement_pct = ((best_zero_shot[1] - best_ft[1]) / best_zero_shot[1]) * 100
        
        print(f"\n📈 ANÁLISE DE MELHORIA:")
        print(f"   Zero-shot melhor: {best_zero_shot[2]:.2f}km")
        print(f"   Fine-tuned melhor: {best_ft[2]:.2f}km")
        print(f"   🎯 Melhoria: {improvement_pct:.1f}% ({best_zero_shot[2]-best_ft[2]:.2f}km)")
        
        # 🔬 MLflow - Log dados para o paper
        if mlflow_tracker and use_mlflow:
            mlflow_tracker.log_paper_summary({
                "final_avg_error_km": best_ft[2],
                "zero_shot_error_km": best_zero_shot[2], 
                "improvement_pct": improvement_pct,
                "n_experiments": len(checkpoints)
            })
    
    # Relatório detalhado por cidade
    print(f"\n📋 DETALHES POR CIDADE:")
    print("=" * 50)
    
    for city in target_cities:
        print(f"\n🏙️ Cidade {city}:")
        city_results = []
        
        for model_name in checkpoints.keys():
            if city in results[model_name]:
                error = results[model_name][city]['cell_error']
                km = results[model_name][city]['distance_km']
                if error != float('inf'):
                    city_results.append((model_name, error, km))
        
        city_results.sort(key=lambda x: x[1])  # Ordena por erro
        
        for i, (model_name, error, km) in enumerate(city_results):
            ranking = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}º"
            print(f"  {ranking} {model_name:20s}: {error:.2f} células ({km:.2f}km)")
    
    return results


def compare_with_baseline_and_generate_submission(
    parquet_path: str,
    device: torch.device = None
):
    """
    Compara modelos e gera submissão com o melhor modelo.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("🎯 COMPARAÇÃO E GERAÇÃO DE SUBMISSÃO")
    print("=" * 50)
    
    # 1. Avalia todos os modelos
    results = evaluate_all_models_with_mlflow(
        parquet_path=parquet_path,
        device=device,
        use_mlflow=True
    )
    
    if not results:
        return None
    
    # 2. Encontra o melhor modelo
    best_model = None
    best_avg_error = float('inf')
    
    for model_name, cities_data in results.items():
        valid_errors = [
            v['cell_error'] for v in cities_data.values() 
            if v['cell_error'] != float('inf')
        ]
        
        if valid_errors:
            avg_error = np.mean(valid_errors)
            if avg_error < best_avg_error:
                best_avg_error = avg_error
                best_model = model_name
    
    if not best_model:
        print("❌ Nenhum modelo válido encontrado")
        return None
    
    print(f"\n🏆 MELHOR MODELO: {best_model}")
    print(f"📈 Erro médio: {best_avg_error:.2f} células ({best_avg_error*0.5:.2f}km)")
    
    # 3. Determina checkpoint do melhor modelo
    if "zero" in best_model.lower():
        best_checkpoint = "humob_model_A.pt"
    else:
        # Extrai cidade do nome (ex: "Fine-tuned D" -> "D")
        city = best_model.split()[-1] 
        best_checkpoint = f"humob_model_finetuned_{city}.pt"
    
    print(f"📁 Checkpoint: {best_checkpoint}")
    
    # 4. Oferece gerar submissão
    response = input(f"\n📄 Gerar submissão com {best_model}? (y/n): ").strip().lower()
    
    if response == 'y':
        print(f"🔄 Gerando submissão com {best_model}...")
        
        output_file = f"humob_best_submission_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        try:
            submission_df = generate_humob_submission(
                parquet_path=parquet_path,
                checkpoint_path=best_checkpoint,
                device=device,
                target_cities=["B", "C", "D"],
                submission_days=(61, 75),
                sequence_length=24,
                output_file=output_file
            )
            
            print(f"✅ Submissão gerada: {output_file}")
            print(f"📊 Predições: {len(submission_df):,}")
            print(f"👥 Usuários únicos: {submission_df['uid'].nunique():,}")
            print(f"🎯 Modelo usado: {best_model} (erro médio: {best_avg_error*0.5:.2f}km)")
            print(f"\n🚀 PRONTO PARA SUBMISSÃO NO HUMOB CHALLENGE!")
            
            return submission_df
            
        except Exception as e:
            print(f"❌ Erro gerando submissão: {e}")
            return None
    
    return results


def mlflow_experiments_dashboard():
    """
    Exibe dashboard dos experimentos para análise acadêmica.
    """
    print("🔬 DASHBOARD DOS EXPERIMENTOS MLFLOW")
    print("=" * 50)
    
    try:
        summary = get_experiment_summary_for_paper()
        
        if summary:
            print("📊 ESTATÍSTICAS DOS EXPERIMENTOS:")
            print(f"   Total de runs: {summary['total_runs']}")
            print(f"   Treinamento base: {summary['base_runs']}")
            print(f"   Fine-tuning: {summary['ft_runs']}")
            print(f"   Comparações: {summary['comparison_runs']}")
            
            if summary['best_val_loss']:
                print(f"   Melhor val loss: {summary['best_val_loss']:.4f}")
            
            print(f"\n📋 DADOS PARA O PAPER:")
            print(f"   - Arquitetura híbrida implementada")
            print(f"   - {summary['ft_runs']} experimentos de fine-tuning")
            print(f"   - {summary['comparison_runs']} comparações realizadas")
            print(f"   - Tracking completo de hiperparâmetros e métricas")
            
            print(f"\n🖥️ PARA VISUALIZAR:")
            print(f"   Execute: mlflow ui --backend-store-uri ./mlruns")
            print(f"   Acesse: http://localhost:5000")
            print(f"   📁 Dados salvos em: ./mlruns/")
            
            print(f"\n📈 GRÁFICOS E PLOTS:")
            print(f"   - Curvas de treinamento")
            print(f"   - Evolução dos pesos da fusão")
            print(f"   - Comparação de modelos")
            print(f"   - Métricas por cidade")
            
        else:
            print("❌ Nenhum experimento MLflow encontrado")
            print("💡 Para gerar dados:")
            print("   1. Execute run_humob.py > Opção 7")
            print("   2. Ou execute este script > Opção 3")
    
    except Exception as e:
        print(f"❌ Erro acessando MLflow: {e}")
        print("💡 Instale o MLflow: pip install mlflow")


def main():
    """Menu principal para avaliação com MLflow."""
    
    # Configurações - AJUSTE AQUI
    parquet_file = "humob_all_cities_v2_normalized.parquet"
    
    print("🔍 AVALIAÇÃO E ANÁLISE - HUMOB COM MLFLOW")
    print("=" * 60)
    print("🔬 Tracking acadêmico completo")
    print(f"📁 Dados: {parquet_file}")
    
    # Verifica arquivos
    if not os.path.exists(parquet_file):
        print(f"❌ Arquivo de dados não encontrado: {parquet_file}")
        return
    
    # Verifica se tem pelo menos um modelo
    has_models = any(
        os.path.exists(f) for f in [
            "humob_model_A.pt",
            "humob_model_finetuned_B.pt", 
            "humob_model_finetuned_C.pt",
            "humob_model_finetuned_D.pt"
        ]
    )
    
    if not has_models:
        print("❌ Nenhum modelo encontrado!")
        print("Execute o treinamento primeiro com run_humob.py")
        return
    
    print("\n🎯 Opções de avaliação:")
    print("1. 📊 Avaliar todos os modelos (com MLflow)")
    print("2. 📄 Gerar submissão com modelo específico")
    print("3. 🏆 Comparar tudo + melhor submissão (RECOMENDADO)")
    print("4. 🔬 Dashboard dos experimentos MLflow")
    print("5. 📈 Avaliar sem MLflow (mais rápido)")
    
    try:
        choice = input("\nEscolha (1-5): ").strip()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {device}")
        
        if choice == "1":
            evaluate_all_models_with_mlflow(
                parquet_path=parquet_file,
                device=device,
                use_mlflow=True
            )
            
        elif choice == "2":
            # Lista modelos disponíveis
            models = {}
            if os.path.exists("humob_model_A.pt"):
                models["A"] = "humob_model_A.pt"
            for city in ["B", "C", "D"]:
                checkpoint = f"humob_model_finetuned_{city}.pt"
                if os.path.exists(checkpoint):
                    models[city] = checkpoint
            
            if not models:
                print("❌ Nenhum modelo encontrado")
                return
            
            print("Modelos disponíveis:")
            for i, (name, path) in enumerate(models.items(), 1):
                print(f"  {i}. {name} ({path})")
            
            try:
                idx = int(input("Escolha o modelo (número): ")) - 1
                selected_model = list(models.items())[idx]
                
                output_file = f"humob_submission_{selected_model[0]}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                
                generate_humob_submission(
                    parquet_path=parquet_file,
                    checkpoint_path=selected_model[1],
                    device=device,
                    output_file=output_file
                )
                
            except (ValueError, IndexError):
                print("❌ Seleção inválida")
            
        elif choice == "3":
            compare_with_baseline_and_generate_submission(
                parquet_path=parquet_file,
                device=device
            )
            
        elif choice == "4":
            mlflow_experiments_dashboard()
            
        elif choice == "5":
            evaluate_all_models_with_mlflow(
                parquet_path=parquet_file,
                device=device,
                use_mlflow=False
            )
            
        else:
            print("❌ Opção inválida")
            
    except KeyboardInterrupt:
        print("\n🛑 Operação cancelada pelo usuário")
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()