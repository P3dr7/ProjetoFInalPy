"""
Testes unitários para o módulo data_loader.

Testa carregamento de dados, interpolação e tratamento de erros.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

import sys
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from src.data_loader import RiskFreeRateLoader


@pytest.fixture
def sample_csv_file():
    """Cria um arquivo CSV temporário para testes."""
    csv_content = '''Date,"1 Mo","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"
12/30/2022,4.12,4.41,4.42,4.69,4.76,4.73,4.41,4.22,3.99,3.96,3.88,4.14,3.97
12/29/2022,4.04,4.39,4.45,4.66,4.73,4.71,4.34,4.16,3.94,3.91,3.83,4.09,3.92
12/28/2022,3.86,4.33,4.46,4.66,4.75,4.71,4.31,4.18,3.97,3.97,3.88,4.13,3.98
12/27/2022,3.87,4.32,4.46,4.66,4.76,4.75,4.32,4.17,3.94,3.93,3.84,4.10,3.93'''
    
    # Cria arquivo temporário
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(csv_content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


class TestRiskFreeRateLoader:
    """Testes para a classe RiskFreeRateLoader."""
    
    def test_initialization(self, sample_csv_file):
        """Testa se o loader inicializa corretamente."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        assert loader.data is not None
        assert len(loader.data) == 4
        assert 'Date' in loader.data.columns
    
    def test_file_not_found(self):
        """Testa erro quando arquivo não existe."""
        with pytest.raises(FileNotFoundError):
            RiskFreeRateLoader('arquivo_inexistente.csv')
    
    def test_date_parsing(self, sample_csv_file):
        """Testa se as datas são parseadas corretamente."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        # Verifica se a coluna Date é datetime
        assert pd.api.types.is_datetime64_any_dtype(loader.data['Date'])
        
        # Verifica primeira data (dados são ordenados, então primeira é 12/27)
        first_date = loader.data['Date'].iloc[0]
        assert first_date == pd.Timestamp('2022-12-27')
    
    def test_rate_conversion(self, sample_csv_file):
        """Testa se as taxas são convertidas de % para decimal."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        # Primeira linha após ordenação: 12/27 com taxa 3.87%, deve ser ~0.0387
        rate_1mo = loader.data['1 Mo'].iloc[0]
        assert np.isclose(rate_1mo, 0.0387, rtol=1e-4)
    
    def test_get_exact_date_and_maturity(self, sample_csv_file):
        """Testa busca com data e maturidade exatas."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        date = datetime(2022, 12, 30)
        rate = loader.get_risk_free_rate(
            date, 
            maturity_days=30,
            interpolate_date=False,
            interpolate_maturity=False
        )
        
        # Deve retornar 0.0412
        assert np.isclose(rate, 0.0412, rtol=1e-4)
    
    def test_maturity_interpolation(self, sample_csv_file):
        """Testa interpolação entre maturidades."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        date = datetime(2022, 12, 30)
        
        # Taxa para 1 Mo (30 dias) e 2 Mo (60 dias)
        rate_30 = loader.get_risk_free_rate(date, 30, interpolate_maturity=False)
        rate_60 = loader.get_risk_free_rate(date, 60, interpolate_maturity=False)
        
        # Taxa interpolada para 45 dias (meio do caminho)
        rate_45 = loader.get_risk_free_rate(date, 45, interpolate_maturity=True)
        
        # Deve estar entre as duas taxas
        assert min(rate_30, rate_60) <= rate_45 <= max(rate_30, rate_60)
    
    def test_date_interpolation(self, sample_csv_file):
        """Testa interpolação entre datas."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        # Data entre 12/29 e 12/30
        date = datetime(2022, 12, 29, 12, 0)  # Meio-dia de 12/29
        
        rate = loader.get_risk_free_rate(
            date,
            maturity_days=30,
            interpolate_date=True,
            interpolate_maturity=False
        )
        
        # Taxa deve estar no range válido
        assert 0 < rate < 1.0
    
    def test_invalid_maturity(self, sample_csv_file):
        """Testa erro com maturidade inválida."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        date = datetime(2022, 12, 30)
        
        with pytest.raises(ValueError, match="Maturidade deve ser maior que zero"):
            loader.get_risk_free_rate(date, maturity_days=0)
        
        with pytest.raises(ValueError, match="Maturidade deve ser maior que zero"):
            loader.get_risk_free_rate(date, maturity_days=-10)
    
    def test_get_available_date_range(self, sample_csv_file):
        """Testa obtenção do range de datas."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        min_date, max_date = loader.get_available_date_range()
        
        assert min_date == pd.Timestamp('2022-12-27')
        assert max_date == pd.Timestamp('2022-12-30')
    
    def test_repr(self, sample_csv_file):
        """Testa representação string."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        repr_str = repr(loader)
        assert "RiskFreeRateLoader" in repr_str
        assert "4 registros" in repr_str
    
    def test_extrapolation(self, sample_csv_file):
        """Testa extrapolação para maturidades muito longas."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        date = datetime(2022, 12, 30)
        
        # Maturidade além de 30 anos (maior disponível)
        rate = loader.get_risk_free_rate(
            date,
            maturity_days=15000,  # ~41 anos
            interpolate_maturity=True
        )
        
        # Deve retornar um valor válido (extrapolado)
        assert 0 < rate < 1.0


class TestInterpolationAccuracy:
    """Testes de precisão da interpolação."""
    
    def test_linear_interpolation_midpoint(self, sample_csv_file):
        """Testa se interpolação linear está correta no ponto médio."""
        loader = RiskFreeRateLoader(sample_csv_file)
        
        date = datetime(2022, 12, 30)
        
        # Pega taxas dos extremos
        rate_30 = loader.get_risk_free_rate(date, 30, interpolate_maturity=False)
        rate_60 = loader.get_risk_free_rate(date, 60, interpolate_maturity=False)
        
        # Interpola no meio
        rate_45 = loader.get_risk_free_rate(date, 45, interpolate_maturity=True)
        
        # Para interpolação linear, deve ser a média
        expected = (rate_30 + rate_60) / 2
        assert np.isclose(rate_45, expected, rtol=1e-3)


def test_real_csv_file():
    """Testa com o arquivo CSV real do projeto."""
    csv_path = Path(__file__).parent.parent / 'databasePy.csv'
    
    if not csv_path.exists():
        pytest.skip("databasePy.csv não encontrado")
    
    loader = RiskFreeRateLoader(str(csv_path))
    
    # Verifica que carregou dados
    assert loader.data is not None
    assert len(loader.data) > 0
    
    # Testa uma operação básica
    min_date, max_date = loader.get_available_date_range()
    rate = loader.get_risk_free_rate(max_date, 90)
    
    assert 0 < rate < 1.0
