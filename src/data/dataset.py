import torch
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from tqdm import tqdm


class HuMobNormalizedDataset(IterableDataset):
    """
    Dataset otimizado para dados JÁ NORMALIZADOS do HuMob.
    Constrói sequências temporais adequadas para a LSTM.
    
    Espera colunas:
    - uid, city_encoded, d_norm, t_sin, t_cos 
    - x_norm, y_norm (coordenadas normalizadas [0,1])
    - POI_norm (array 85-D normalizado)
    """
    def __init__(
        self,
        parquet_path: str,
        cities: list[str] = ["A"],
        mode: str = "train",  # "train", "val", "test"
        sequence_length: int = 24,  # CORRIGIDO: agora > 1 para LSTM fazer sentido
        prediction_steps: int = 1,
        chunk_size: int = 10_000,
        max_sequences_per_user: int = 50,
        train_days: tuple = (0.0, 0.8),      # Normalizado [0,1]
        val_days: tuple = (0.8, 1.0),
        test_days: tuple = (0.8, 1.0)       # Para cidades B,C,D
    ):
        self.parquet_path = parquet_path
        self.cities = cities
        self.cities_set = set(cities)
        self.mode = mode
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.chunk_size = chunk_size
        self.max_sequences_per_user = max_sequences_per_user
        
        # Define range de dias baseado no modo
        if mode == "train":
            self.day_range = train_days
        elif mode == "val":
            self.day_range = val_days
        else:  # test
            self.day_range = test_days

    def _build_temporal_sequences(self, user_data):
        """
        Retorna sequências de tamanho `sequence_length` para alimentar a LSTM.
        """
        sequences = []
        user_data = user_data.sort_values(['d_norm', 't_sin', 't_cos'])
        
        # Precisa ter pelo menos sequence_length + prediction_steps pontos
        min_length = self.sequence_length + self.prediction_steps
        if len(user_data) < min_length:
            return sequences
            
        for i in range(len(user_data) - min_length + 1):
            # Janela de entrada: i até i+sequence_length
            seq_data = user_data.iloc[i:i+self.sequence_length]
            
            # Alvo: próximos prediction_steps pontos
            target_data = user_data.iloc[
                i+self.sequence_length:i+self.sequence_length+self.prediction_steps
            ]
            
            # Extrai sequência de coordenadas normalizadas
            coords_seq = seq_data[['x_norm', 'y_norm']].values.astype(np.float32)
            target_coords = target_data[['x_norm', 'y_norm']].values.astype(np.float32)
            
            # Informação do último ponto da sequência (para contexto)
            last_point = seq_data.iloc[-1]
            
            sequences.append({
                'uid': int(last_point['uid']),
                'd_norm': float(last_point['d_norm']),
                't_sin': float(last_point['t_sin']),
                't_cos': float(last_point['t_cos']),
                'city_encoded': int(last_point['city_encoded']),
                'poi_norm': last_point['POI_norm'].astype(np.float32),
                'coords_seq': coords_seq,              # (sequence_length, 2)
                'target_coords': target_coords         # (prediction_steps, 2)
            })
            
        return sequences

    def _sample_user_sequences(self, user_group, max_seqs: int):
        """Amostra sequências de um usuário de forma estratificada."""
        sequences = self._build_temporal_sequences(user_group)
        
        if len(sequences) <= max_seqs:
            return sequences
        
        # Amostragem estratificada por período do dia
        sequences_by_period = {}
        for seq in sequences:
            # Converte t_sin/t_cos de volta para período aproximado
            t_raw = np.arctan2(seq['t_sin'], seq['t_cos']) / (2 * np.pi) * 48
            t_raw = int(t_raw % 48)
            period = t_raw // 6  # 8 períodos de 6 horas
            
            if period not in sequences_by_period:
                sequences_by_period[period] = []
            sequences_by_period[period].append(seq)
        
        # Amostra proporcionalmente
        sampled = []
        for period_seqs in sequences_by_period.values():
            n_sample = max(1, min(len(period_seqs), max_seqs // len(sequences_by_period)))
            if len(period_seqs) >= n_sample:
                sampled.extend(np.random.choice(period_seqs, n_sample, replace=False))
            else:
                sampled.extend(period_seqs)
        
        return sampled[:max_seqs]

    def check_data_sanity(self, df):
        """Verifica se dados normalizados estão OK."""
        issues = []
        
        # Verifica coordenadas normalizadas [0,1]
        for coord in ['x_norm', 'y_norm']:
            if coord in df.columns:
                coord_min, coord_max = df[coord].min(), df[coord].max()
                if coord_min < -0.1 or coord_max > 1.1:  # Pequena tolerância
                    issues.append(f"{coord} fora do range [0,1]: [{coord_min:.3f}, {coord_max:.3f}]")
        
        # Verifica d_norm [0,1]
        if 'd_norm' in df.columns:
            d_min, d_max = df['d_norm'].min(), df['d_norm'].max()
            if d_min < -0.1 or d_max > 1.1:
                issues.append(f"d_norm fora do range [0,1]: [{d_min:.3f}, {d_max:.3f}]")
        
        # Verifica t_sin, t_cos [-1,1]
        for t_col in ['t_sin', 't_cos']:
            if t_col in df.columns:
                t_min, t_max = df[t_col].min(), df[t_col].max()
                if t_min < -1.1 or t_max > 1.1:
                    issues.append(f"{t_col} fora do range [-1,1]: [{t_min:.3f}, {t_max:.3f}]")
        
        if issues:
            print(f"⚠️ Encontrados {len(issues)} problemas:")
            for issue in issues[:3]:
                print(f"   • {issue}")
            return False
        
        return True

    def __iter__(self):
        pf = pq.ParquetFile(self.parquet_path)
        
        for batch in pf.iter_batches(batch_size=self.chunk_size):
            table = pa.Table.from_batches([batch], schema=pf.schema_arrow)

            # Filtra cidades
            if 'city' in table.column_names:
                city_mask = pc.is_in(table.column('city'), pa.array(list(self.cities_set)))
            elif 'city_encoded' in table.column_names:
                # Se já tem city_encoded, filtra por valores numéricos
                city_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                target_cities = [city_map[c] for c in self.cities if c in city_map]
                city_mask = pc.is_in(table.column('city_encoded'), pa.array(target_cities))
            else:
                continue
                
            table = table.filter(city_mask)
            if table.num_rows == 0:
                continue

            # Filtra por range de dias (normalizado)
            if 'd_norm' in table.column_names:
                day_mask = pc.and_(
                    pc.greater_equal(table.column("d_norm"), self.day_range[0]),
                    pc.less_equal(table.column("d_norm"), self.day_range[1])
                )
                table = table.filter(day_mask)
                if table.num_rows == 0:
                    continue

            # Converte para pandas
            df = table.to_pandas()
            
            # Verifica sanidade dos dados
            if not self.check_data_sanity(df):
                print("⚠️ Pulando batch com problemas nos dados")
                continue
            
            # Processa cada usuário
            for uid, user_group in df.groupby('uid'):
                if len(user_group) < self.sequence_length + self.prediction_steps:
                    continue
                
                sequences = self._sample_user_sequences(
                    user_group, 
                    self.max_sequences_per_user
                )
                
                for seq in sequences:
                    yield (
                        torch.tensor(seq['uid'], dtype=torch.long),
                        torch.tensor(seq['d_norm'], dtype=torch.float32),
                        torch.tensor(seq['t_sin'], dtype=torch.float32),
                        torch.tensor(seq['t_cos'], dtype=torch.float32),
                        torch.tensor(seq['city_encoded'], dtype=torch.long),
                        torch.from_numpy(seq['poi_norm']),
                        torch.from_numpy(seq['coords_seq']),
                        torch.from_numpy(seq['target_coords'])
                    )


def create_humob_loaders(
    parquet_path: str,
    cities: list[str] = ["A"],
    batch_size: int = 32,
    sequence_length: int = 24,
    max_sequences_per_user: int = 50,
    num_workers: int = 0
):
    """Cria loaders de treino e validação."""
    
    train_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=cities,
        mode="train",
        sequence_length=sequence_length,
        max_sequences_per_user=max_sequences_per_user
    )
    
    val_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=cities,
        mode="val", 
        sequence_length=sequence_length,
        max_sequences_per_user=max_sequences_per_user // 2
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers, 
        pin_memory=False
    )
    
    return train_loader, val_loader


def create_test_loader(
    parquet_path: str,
    cities: list[str] = ["B", "C", "D"],
    batch_size: int = 32,
    sequence_length: int = 24,
    num_workers: int = 0
):
    """Cria loader para teste (avaliação final)."""
    
    test_ds = HuMobNormalizedDataset(
        parquet_path=parquet_path,
        cities=cities,
        mode="test",
        sequence_length=sequence_length,
        max_sequences_per_user=1000  # Mais sequências para teste
    )
    
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False
    )