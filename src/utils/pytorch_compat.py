"""
Utilitários para compatibilidade PyTorch 2.6+.
Resolve o problema de carregamento de checkpoints que apareceu nos logs.

PROBLEMA IDENTIFICADO:
PyTorch 2.6+ mudou o padrão de `weights_only=False` para `weights_only=True`,
causando erro ao carregar numpy arrays salvos nos checkpoints.
"""

import torch
import numpy as np
import warnings
from pathlib import Path


def load_checkpoint_safe(checkpoint_path: str, device: torch.device = None, verbose: bool = True):
    """
    Carrega checkpoint com total compatibilidade PyTorch 2.6+.
    
    Analogia: Como um "tradutor universal" que consegue ler checkpoints
    salvos em diferentes versões do PyTorch.
    
    Args:
        checkpoint_path: Caminho do checkpoint
        device: Device para carregar (padrão: CPU)
        verbose: Se deve imprimir logs de debug
        
    Returns:
        dict: Checkpoint carregado
        
    Raises:
        FileNotFoundError: Se arquivo não existe
        RuntimeError: Se não conseguir carregar o checkpoint
    """
    if device is None:
        device = torch.device("cpu")
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")
    
    if verbose:
        print(f"📂 Carregando checkpoint: {checkpoint_path}")
    
    # Estratégia 1: Método padrão (PyTorch < 2.6)
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if verbose:
            print("✅ Carregado com método padrão (weights_only=False)")
        return ckpt
        
    except Exception as e1:
        if verbose:
            print(f"⚠️ Método padrão falhou: {type(e1).__name__}")
        
        # Estratégia 2: Safe globals (PyTorch 2.6+)
        try:
            with torch.serialization.safe_globals([
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                np.core.multiarray.scalar,
                np.core._multiarray_umath._reconstruct
            ]):
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
                if verbose:
                    print("✅ Carregado com safe_globals (PyTorch 2.6+)")
                return ckpt
                
        except Exception as e2:
            if verbose:
                print(f"⚠️ Safe globals falhou: {type(e2).__name__}")
            
            # Estratégia 3: Pickle protocol direto
            try:
                import pickle
                with open(checkpoint_path, 'rb') as f:
                    ckpt = pickle.load(f)
                    
                # Move tensors para device correto
                if isinstance(ckpt, dict):
                    for key, value in ckpt.items():
                        if isinstance(value, torch.Tensor):
                            ckpt[key] = value.to(device)
                        elif isinstance(value, np.ndarray):
                            # Converte numpy para tensor
                            ckpt[key] = torch.from_numpy(value).to(device)
                
                if verbose:
                    print("✅ Carregado com pickle direto")
                return ckpt
                
            except Exception as e3:
                if verbose:
                    print(f"❌ Todas as estratégias falharam:")
                    print(f"   1. Padrão: {e1}")
                    print(f"   2. Safe globals: {e2}")
                    print(f"   3. Pickle: {e3}")
                
                raise RuntimeError(
                    f"Não foi possível carregar checkpoint {checkpoint_path}. "
                    f"Tente recriar o checkpoint ou use uma versão anterior do PyTorch."
                ) from e3


def save_checkpoint_compatible(
    checkpoint_dict: dict,
    checkpoint_path: str,
    pytorch_version: str = None,
    verbose: bool = True
):
    """
    Salva checkpoint de forma compatível com diferentes versões PyTorch.
    
    Args:
        checkpoint_dict: Dicionário com dados do checkpoint
        checkpoint_path: Onde salvar
        pytorch_version: Versão do PyTorch (auto-detecta se None)
        verbose: Se deve imprimir logs
    """
    checkpoint_path = Path(checkpoint_path)
    
    if pytorch_version is None:
        pytorch_version = torch.__version__
    
    if verbose:
        print(f"💾 Salvando checkpoint: {checkpoint_path}")
        print(f"🔧 PyTorch version: {pytorch_version}")
    
    # Converte numpy arrays para tensors para compatibilidade
    safe_checkpoint = {}
    
    for key, value in checkpoint_dict.items():
        if isinstance(value, np.ndarray):
            # Converte numpy para tensor e volta para numpy
            # Isso garante serialização compatível
            tensor_value = torch.from_numpy(value)
            safe_checkpoint[key] = tensor_value.detach().cpu().numpy()
        else:
            safe_checkpoint[key] = value
    
    # Adiciona metadados de compatibilidade
    safe_checkpoint['_pytorch_version'] = pytorch_version
    safe_checkpoint['_save_timestamp'] = torch.tensor(pd.Timestamp.now().timestamp())
    
    try:
        # Para PyTorch 2.6+, usa protocolo pickle compatível
        torch.save(safe_checkpoint, checkpoint_path, pickle_protocol=4)
        
        if verbose:
            print("✅ Checkpoint salvo com sucesso")
            
    except Exception as e:
        print(f"❌ Erro salvando checkpoint: {e}")
        raise


def check_pytorch_version_compatibility(verbose: bool = True):
    """
    Verifica compatibilidade da versão do PyTorch e sugere correções.
    
    Returns:
        dict: Informações sobre compatibilidade
    """
    import torch
    version = torch.__version__
    major, minor = map(int, version.split('.')[:2])
    
    info = {
        'version': version,
        'major': major,
        'minor': minor,
        'is_2_6_plus': major >= 2 and minor >= 6,
        'needs_safe_loading': major >= 2 and minor >= 6,
        'supports_weights_only': major >= 2 and minor >= 5
    }
    
    if verbose:
        print(f"🔧 COMPATIBILIDADE PYTORCH")
        print(f"   Versão: {version}")
        print(f"   É 2.6+? {'✅' if info['is_2_6_plus'] else '❌'}")
        print(f"   Precisa safe loading? {'✅' if info['needs_safe_loading'] else '❌'}")
        
        if info['needs_safe_loading']:
            print(f"\n💡 DICAS PARA PYTORCH 2.6+:")
            print(f"   - Use load_checkpoint_safe() em vez de torch.load()")
            print(f"   - Checkpoints antigos podem precisar ser recriados")
            print(f"   - Use save_checkpoint_compatible() para novos checkpoints")
    
    return info


def fix_existing_checkpoints(checkpoint_dir: str = ".", verbose: bool = True):
    """
    Corrige checkpoints existentes para compatibilidade PyTorch 2.6+.
    
    Args:
        checkpoint_dir: Diretório com checkpoints (.pt files)
        verbose: Se deve imprimir logs detalhados
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Busca todos os arquivos .pt
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        if verbose:
            print("❌ Nenhum checkpoint encontrado")
        return
    
    if verbose:
        print(f"🔄 CORRIGINDO {len(checkpoints)} CHECKPOINTS")
        print(f"📁 Diretório: {checkpoint_dir}")
    
    device = torch.device("cpu")  # Usa CPU para evitar problemas de memória
    
    for checkpoint_path in checkpoints:
        if verbose:
            print(f"\n🔧 Processando: {checkpoint_path.name}")
        
        try:
            # Tenta carregar com método seguro
            ckpt = load_checkpoint_safe(checkpoint_path, device, verbose=False)
            
            # Cria backup
            backup_path = checkpoint_path.with_suffix('.pt.backup')
            if not backup_path.exists():
                checkpoint_path.rename(backup_path)
                if verbose:
                    print(f"📄 Backup criado: {backup_path.name}")
            
            # Salva versão compatível
            save_checkpoint_compatible(ckpt, checkpoint_path, verbose=False)
            
            if verbose:
                print(f"✅ Checkpoint corrigido: {checkpoint_path.name}")
                
        except Exception as e:
            if verbose:
                print(f"❌ Erro corrigindo {checkpoint_path.name}: {e}")


def test_checkpoint_loading():
    """
    Testa se o carregamento de checkpoints está funcionando.
    """
    print("🧪 TESTE DE COMPATIBILIDADE DE CHECKPOINTS")
    print("=" * 50)
    
    # Verifica versão PyTorch
    info = check_pytorch_version_compatibility()
    
    # Busca checkpoints para testar
    checkpoints = []
    for pattern in ["*.pt", "humob_model_*.pt"]:
        checkpoints.extend(Path(".").glob(pattern))
    
    if not checkpoints:
        print("❌ Nenhum checkpoint encontrado para testar")
        print("💡 Execute o treinamento primeiro para gerar checkpoints")
        return
    
    print(f"\n🔍 Testando {len(checkpoints)} checkpoints...")
    
    success_count = 0
    
    for checkpoint_path in checkpoints:
        print(f"\n📂 Testando: {checkpoint_path.name}")
        
        try:
            ckpt = load_checkpoint_safe(checkpoint_path, verbose=False)
            
            # Verifica estrutura básica
            required_keys = ['state_dict', 'config']
            missing_keys = [k for k in required_keys if k not in ckpt]
            
            if missing_keys:
                print(f"⚠️ Chaves ausentes: {missing_keys}")
            else:
                print(f"✅ Estrutura OK")
            
            # Verifica se centers existe
            if 'centers' in ckpt:
                centers_shape = ckpt['centers'].shape if hasattr(ckpt['centers'], 'shape') else "Unknown"
                print(f"📍 Centers: {centers_shape}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Falha: {e}")
    
    print(f"\n📊 RESULTADO: {success_count}/{len(checkpoints)} checkpoints carregados com sucesso")
    
    if success_count < len(checkpoints):
        print("💡 Para corrigir checkpoints problemáticos:")
        print("   fix_existing_checkpoints()")


if __name__ == "__main__":
    # Executa testes de compatibilidade
    test_checkpoint_loading()