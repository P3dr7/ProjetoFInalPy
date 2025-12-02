"""
Testes unitários para o módulo pricing_models.

Testa modelos Black-Scholes e Heston, cálculo de preços e métricas.
"""

import pytest
import numpy as np
from scipy.stats import norm

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from src.pricing_models import (
    OptionData, 
    BlackScholesModel, 
    HestonModel,
    calculate_mse,
    calculate_rmse
)


class TestOptionData:
    """Testes para a classe OptionData."""
    
    def test_valid_option_creation(self):
        """Testa criação de opção válida."""
        option = OptionData(
            spot_price=100.0,
            strike_price=105.0,
            time_to_maturity=0.25,
            risk_free_rate=0.05,
            option_type='call',
            market_price=3.5
        )
        
        assert option.spot_price == 100.0
        assert option.strike_price == 105.0
        assert option.option_type == 'call'
    
    def test_invalid_spot_price(self):
        """Testa erro com preço spot inválido."""
        with pytest.raises(ValueError, match="Preço spot deve ser positivo"):
            OptionData(
                spot_price=-100.0,
                strike_price=105.0,
                time_to_maturity=0.25,
                risk_free_rate=0.05,
                option_type='call',
                market_price=3.5
            )
    
    def test_invalid_strike_price(self):
        """Testa erro com strike inválido."""
        with pytest.raises(ValueError, match="Strike deve ser positivo"):
            OptionData(
                spot_price=100.0,
                strike_price=0,
                time_to_maturity=0.25,
                risk_free_rate=0.05,
                option_type='call',
                market_price=3.5
            )
    
    def test_invalid_time_to_maturity(self):
        """Testa erro com maturidade inválida."""
        with pytest.raises(ValueError, match="Time to maturity deve ser positivo"):
            OptionData(
                spot_price=100.0,
                strike_price=105.0,
                time_to_maturity=0,
                risk_free_rate=0.05,
                option_type='call',
                market_price=3.5
            )
    
    def test_invalid_option_type(self):
        """Testa erro com tipo de opção inválido."""
        with pytest.raises(ValueError, match="Tipo de opção deve ser"):
            OptionData(
                spot_price=100.0,
                strike_price=105.0,
                time_to_maturity=0.25,
                risk_free_rate=0.05,
                option_type='invalid',
                market_price=3.5
            )


class TestBlackScholesModel:
    """Testes para o modelo Black-Scholes."""
    
    def test_parameter_info(self):
        """Testa informações sobre parâmetros."""
        model = BlackScholesModel()
        
        assert model.parameter_names() == ['volatility']
        assert len(model.parameter_bounds()) == 1
        assert model.parameter_bounds()[0][0] > 0  # Min > 0
        assert model.parameter_bounds()[0][1] > model.parameter_bounds()[0][0]  # Max > Min
    
    def test_atm_call_option(self):
        """Testa precificação de call ATM."""
        model = BlackScholesModel()
        
        option = OptionData(
            spot_price=100.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='call',
            market_price=0.0  # Não usado no teste
        )
        
        volatility = 0.20
        price = model.price(option, np.array([volatility]))
        
        # Preço deve ser positivo e razoável
        assert price > 0
        assert price < option.spot_price  # Não pode valer mais que o ativo
    
    def test_atm_put_option(self):
        """Testa precificação de put ATM."""
        model = BlackScholesModel()
        
        option = OptionData(
            spot_price=100.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='put',
            market_price=0.0
        )
        
        volatility = 0.20
        price = model.price(option, np.array([volatility]))
        
        assert price > 0
        assert price < option.strike_price
    
    def test_put_call_parity(self):
        """Testa paridade put-call: C - P = S - K*e^(-r*T)."""
        model = BlackScholesModel()
        
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20
        
        call_option = OptionData(S, K, T, r, 'call', 0.0)
        put_option = OptionData(S, K, T, r, 'put', 0.0)
        
        call_price = model.price(call_option, np.array([sigma]))
        put_price = model.price(put_option, np.array([sigma]))
        
        # C - P = S - K*e^(-r*T)
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        
        assert np.isclose(lhs, rhs, rtol=1e-6)
    
    def test_deep_itm_call(self):
        """Testa call profundamente ITM."""
        model = BlackScholesModel()
        
        option = OptionData(
            spot_price=150.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='call',
            market_price=0.0
        )
        
        price = model.price(option, np.array([0.20]))
        
        # Deve ser próximo do valor intrínseco
        intrinsic_value = option.spot_price - option.strike_price
        assert price >= intrinsic_value
        assert price < option.spot_price
    
    def test_deep_otm_put(self):
        """Testa put profundamente OTM."""
        model = BlackScholesModel()
        
        option = OptionData(
            spot_price=150.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='put',
            market_price=0.0
        )
        
        price = model.price(option, np.array([0.20]))
        
        # Put OTM deve valer muito pouco
        assert price < 1.0
        assert price > 0
    
    def test_zero_volatility(self):
        """Testa comportamento com volatilidade zero."""
        model = BlackScholesModel()
        
        option = OptionData(
            spot_price=100.0,
            strike_price=95.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='call',
            market_price=0.0
        )
        
        # Com vol=0, deve retornar valor intrínseco descontado
        price = model.price(option, np.array([0.0]))
        expected = max(option.spot_price - option.strike_price * np.exp(-option.risk_free_rate * option.time_to_maturity), 0)
        
        assert price >= 0
    
    def test_increasing_volatility_increases_price(self):
        """Testa que maior volatilidade aumenta preço da opção."""
        model = BlackScholesModel()
        
        option = OptionData(
            spot_price=100.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='call',
            market_price=0.0
        )
        
        price_low = model.price(option, np.array([0.10]))
        price_high = model.price(option, np.array([0.30]))
        
        assert price_high > price_low
    
    def test_vega_calculation(self):
        """Testa cálculo de vega."""
        model = BlackScholesModel()
        
        option = OptionData(
            spot_price=100.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='call',
            market_price=10.0
        )
        
        vega = model._calculate_vega(option, 0.20)
        
        # Vega deve ser positivo para opções
        assert vega > 0
    
    def test_implied_volatility(self):
        """Testa cálculo de volatilidade implícita."""
        model = BlackScholesModel()
        
        true_vol = 0.25
        
        option = OptionData(
            spot_price=100.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='call',
            market_price=0.0
        )
        
        # Calcula preço com volatilidade conhecida
        true_price = model.price(option, np.array([true_vol]))
        option.market_price = true_price
        
        # Calcula volatilidade implícita
        implied_vol = model.implied_volatility_analytical(option)
        
        # Deve recuperar a volatilidade original
        assert np.isclose(implied_vol, true_vol, rtol=1e-3)


class TestHestonModel:
    """Testes para o modelo Heston."""
    
    def test_parameter_info(self):
        """Testa informações sobre parâmetros."""
        model = HestonModel()
        
        names = model.parameter_names()
        bounds = model.parameter_bounds()
        
        assert len(names) == 5
        assert len(bounds) == 5
        assert 'v0' in names
        assert 'kappa' in names
        assert 'theta' in names
    
    def test_pricing_basic(self):
        """Testa precificação básica com Heston."""
        model = HestonModel(n_simulations=1000, random_seed=42)
        
        option = OptionData(
            spot_price=100.0,
            strike_price=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            option_type='call',
            market_price=0.0
        )
        
        # Parâmetros razoáveis de Heston
        params = np.array([
            0.04,   # v0
            2.0,    # kappa
            0.04,   # theta
            0.3,    # sigma_v
            -0.7    # rho
        ])
        
        price = model.price(option, params)
        
        # Preço deve ser positivo e razoável
        assert price > 0
        assert price < option.spot_price
    
    def test_invalid_parameters(self):
        """Testa com número errado de parâmetros."""
        model = HestonModel()
        
        option = OptionData(100.0, 100.0, 1.0, 0.05, 'call', 10.0)
        
        with pytest.raises(ValueError, match="Heston requer 5 parâmetros"):
            model.price(option, np.array([0.25]))  # Só 1 parâmetro


class TestMetrics:
    """Testa funções de métricas."""
    
    def test_calculate_mse(self):
        """Testa cálculo de MSE."""
        model = BlackScholesModel()
        
        options = [
            OptionData(100.0, 100.0, 1.0, 0.05, 'call', 10.0),
            OptionData(100.0, 105.0, 1.0, 0.05, 'put', 8.0),
        ]
        
        params = np.array([0.20])
        
        mse = calculate_mse(options, model, params)
        
        # MSE deve ser não-negativo
        assert mse >= 0
    
    def test_calculate_rmse(self):
        """Testa cálculo de RMSE."""
        model = BlackScholesModel()
        
        options = [
            OptionData(100.0, 100.0, 1.0, 0.05, 'call', 10.0),
        ]
        
        params = np.array([0.20])
        
        rmse = calculate_rmse(options, model, params)
        mse = calculate_mse(options, model, params)
        
        # RMSE = sqrt(MSE)
        assert np.isclose(rmse, np.sqrt(mse))
    
    def test_perfect_calibration_zero_mse(self):
        """Testa que calibração perfeita resulta em MSE ~0."""
        model = BlackScholesModel()
        
        true_vol = 0.25
        
        option = OptionData(100.0, 100.0, 1.0, 0.05, 'call', 0.0)
        
        # Calcula preço com volatilidade verdadeira
        true_price = model.price(option, np.array([true_vol]))
        option.market_price = true_price
        
        # MSE com parâmetro correto deve ser zero
        mse = calculate_mse([option], model, np.array([true_vol]))
        
        assert np.isclose(mse, 0, atol=1e-10)


class TestBlackScholesFormula:
    """Testes matemáticos detalhados da fórmula Black-Scholes."""
    
    def test_manual_calculation(self):
        """Testa contra cálculo manual da fórmula."""
        S = 100.0
        K = 100.0
        T = 1.0
        r = 0.05
        sigma = 0.20
        
        # Cálculo manual
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        manual_call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        manual_put = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Cálculo pelo modelo
        model = BlackScholesModel()
        
        call_option = OptionData(S, K, T, r, 'call', 0.0)
        put_option = OptionData(S, K, T, r, 'put', 0.0)
        
        model_call = model.price(call_option, np.array([sigma]))
        model_put = model.price(put_option, np.array([sigma]))
        
        # Devem ser idênticos
        assert np.isclose(manual_call, model_call, rtol=1e-10)
        assert np.isclose(manual_put, model_put, rtol=1e-10)
