"""
Módulo para carregamento e interpolação de dados de taxa de juros.

Este módulo carrega a curva de juros livre de risco (Risk-Free Rate) de um arquivo CSV
e fornece funcionalidades de interpolação para obter taxas em qualquer data/maturidade.
"""

from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


class RiskFreeRateLoader:
    """Carrega e interpola taxas de juros livre de risco."""

    # Mapeamento de colunas para dias de maturidade
    MATURITY_MAPPING: Dict[str, int] = {
        "1 Mo": 30,
        "2 Mo": 60,
        "3 Mo": 90,
        "4 Mo": 120,
        "6 Mo": 180,
        "1 Yr": 365,
        "2 Yr": 730,
        "3 Yr": 1095,
        "5 Yr": 1825,
        "7 Yr": 2555,
        "10 Yr": 3650,
        "20 Yr": 7300,
        "30 Yr": 10950,
    }

    def __init__(self, csv_path: str):
        """
        Inicializa o carregador de taxas de juros.

        Args:
            csv_path: Caminho para o arquivo CSV contendo as taxas de juros.

        Raises:
            FileNotFoundError: Se o arquivo CSV não for encontrado.
            ValueError: Se o arquivo CSV estiver em formato inválido.
        """
        self.csv_path = csv_path
        self.data: Optional[pd.DataFrame] = None
        self._load_data()

    def _load_data(self) -> None:
        """
        Carrega os dados do arquivo CSV.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            ValueError: Se o CSV estiver mal formatado.
        """
        try:
            # Lê o CSV - o arquivo tem aspas extras nas colunas
            self.data = pd.read_csv(self.csv_path)
            
            # Remove aspas extras dos nomes das colunas
            self.data.columns = self.data.columns.str.replace('"', '').str.strip()
            
            # Converte a coluna Date para datetime
            self.data['Date'] = pd.to_datetime(self.data['Date'], format='%m/%d/%Y')
            
            # Ordena por data
            self.data = self.data.sort_values('Date').reset_index(drop=True)
            
            # Verifica se todas as colunas necessárias existem
            expected_cols = ['Date'] + list(self.MATURITY_MAPPING.keys())
            missing_cols = set(expected_cols) - set(self.data.columns)
            if missing_cols:
                raise ValueError(f"Colunas faltando no CSV: {missing_cols}")
            
            # Converte taxas de porcentagem para decimal (4.12% -> 0.0412)
            for col in self.MATURITY_MAPPING.keys():
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce') / 100.0
            
            # Verifica dados faltantes
            if self.data.isnull().any().any():
                print("Aviso: Dados faltantes detectados. Aplicando interpolação...")
                self._interpolate_missing_data()
                
        except FileNotFoundError:
            raise FileNotFoundError(f"Arquivo CSV não encontrado: {self.csv_path}")
        except Exception as e:
            raise ValueError(f"Erro ao carregar CSV: {str(e)}")

    def _interpolate_missing_data(self) -> None:
        """Interpola dados faltantes no dataframe."""
        # Interpolação temporal (entre datas)
        numeric_cols = list(self.MATURITY_MAPPING.keys())
        self.data[numeric_cols] = self.data[numeric_cols].interpolate(
            method='linear', limit_direction='both', axis=0
        )
        
        # Se ainda houver NaN, preenche com forward fill e backward fill
        self.data[numeric_cols] = self.data[numeric_cols].ffill().bfill()

    def get_risk_free_rate(
        self, 
        date: datetime, 
        maturity_days: int,
        interpolate_date: bool = True,
        interpolate_maturity: bool = True
    ) -> float:
        """
        Obtém a taxa de juros livre de risco para uma data e maturidade específicas.

        Args:
            date: Data da opção.
            maturity_days: Dias até a maturidade da opção.
            interpolate_date: Se True, interpola entre datas se a data exata não existir.
            interpolate_maturity: Se True, interpola entre maturidades.

        Returns:
            Taxa de juros livre de risco (decimal, não porcentagem).

        Raises:
            ValueError: Se a data estiver fora do range ou parâmetros inválidos.
        """
        if self.data is None:
            raise ValueError("Dados não carregados.")
        
        if maturity_days <= 0:
            raise ValueError("Maturidade deve ser maior que zero.")

        # Encontra a data mais próxima
        date_row = self._get_date_row(date, interpolate_date)
        
        # Interpola a taxa para a maturidade desejada
        rate = self._interpolate_maturity(date_row, maturity_days, interpolate_maturity)
        
        return rate

    def _get_date_row(self, date: datetime, interpolate: bool) -> pd.Series:
        """
        Obtém a linha de dados para uma data específica.

        Args:
            date: Data desejada.
            interpolate: Se True, interpola entre datas próximas.

        Returns:
            Série com as taxas para diferentes maturidades.
        """
        # Verifica se a data exata existe
        exact_match = self.data[self.data['Date'] == date]
        
        if not exact_match.empty:
            return exact_match.iloc[0]
        
        if not interpolate:
            # Retorna a data mais próxima
            idx = (self.data['Date'] - date).abs().idxmin()
            return self.data.loc[idx]
        
        # Interpolação entre datas
        before = self.data[self.data['Date'] <= date]
        after = self.data[self.data['Date'] > date]
        
        if before.empty or after.empty:
            # Fora do range - retorna a data mais próxima
            idx = (self.data['Date'] - date).abs().idxmin()
            return self.data.loc[idx]
        
        date_before = before.iloc[-1]
        date_after = after.iloc[0]
        
        # Interpolação linear temporal
        delta_total = (date_after['Date'] - date_before['Date']).days
        delta_target = (date - date_before['Date']).days
        weight = delta_target / delta_total if delta_total > 0 else 0
        
        # Interpola todas as colunas de taxa
        interpolated_row = {}
        for col in self.MATURITY_MAPPING.keys():
            interpolated_row[col] = (
                date_before[col] * (1 - weight) + date_after[col] * weight
            )
        
        return pd.Series(interpolated_row)

    def _interpolate_maturity(
        self, 
        date_row: pd.Series, 
        maturity_days: int,
        interpolate: bool
    ) -> float:
        """
        Interpola a taxa para uma maturidade específica.

        Args:
            date_row: Linha de dados com taxas para diferentes maturidades.
            maturity_days: Maturidade desejada em dias.
            interpolate: Se True, interpola entre maturidades.

        Returns:
            Taxa interpolada.
        """
        # Extrai maturidades e taxas disponíveis
        maturities = np.array(list(self.MATURITY_MAPPING.values()))
        rates = np.array([date_row[col] for col in self.MATURITY_MAPPING.keys()])
        
        # Verifica se existe match exato
        if maturity_days in maturities:
            idx = np.where(maturities == maturity_days)[0][0]
            return rates[idx]
        
        if not interpolate:
            # Retorna a maturidade mais próxima
            idx = np.abs(maturities - maturity_days).argmin()
            return rates[idx]
        
        # Interpolação/extrapolação usando scipy
        # Usa interpolação linear com extrapolação
        interp_func = interp1d(
            maturities, 
            rates, 
            kind='linear',
            fill_value='extrapolate'
        )
        
        return float(interp_func(maturity_days))

    def get_available_date_range(self) -> Tuple[datetime, datetime]:
        """
        Retorna o intervalo de datas disponível nos dados.

        Returns:
            Tupla (data_inicial, data_final).
        """
        if self.data is None:
            raise ValueError("Dados não carregados.")
        
        return (self.data['Date'].min(), self.data['Date'].max())

    def __repr__(self) -> str:
        """Representação string do objeto."""
        if self.data is None:
            return "RiskFreeRateLoader(não carregado)"
        
        min_date, max_date = self.get_available_date_range()
        return (
            f"RiskFreeRateLoader({len(self.data)} registros, "
            f"{min_date.date()} a {max_date.date()})"
        )
