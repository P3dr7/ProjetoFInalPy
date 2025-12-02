"""
Módulo de modelos de precificação de opções.

Implementa modelos de precificação incluindo Black-Scholes e estrutura para Heston.
"""

from typing import Protocol, List, Dict
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod


@dataclass
class OptionData:
    """Dados de uma opção financeira."""
    
    spot_price: float          # S: Preço atual do ativo subjacente
    strike_price: float        # K: Preço de exercício
    time_to_maturity: float    # T: Tempo até vencimento (em anos)
    risk_free_rate: float      # r: Taxa livre de risco
    option_type: str           # 'call' ou 'put'
    market_price: float        # Preço de mercado observado
    
    def __post_init__(self):
        """Valida os dados da opção."""
        if self.spot_price <= 0:
            raise ValueError("Preço spot deve ser positivo")
        if self.strike_price <= 0:
            raise ValueError("Strike deve ser positivo")
        if self.time_to_maturity <= 0:
            raise ValueError("Time to maturity deve ser positivo")
        if self.option_type not in ['call', 'put']:
            raise ValueError("Tipo de opção deve ser 'call' ou 'put'")


class PricingModel(ABC):
    """Classe abstrata para modelos de precificação."""
    
    @abstractmethod
    def price(self, option: OptionData, parameters: np.ndarray) -> float:
        """
        Calcula o preço teórico da opção.
        
        Args:
            option: Dados da opção.
            parameters: Vetor de parâmetros do modelo.
            
        Returns:
            Preço teórico da opção.
        """
        pass
    
    @abstractmethod
    def parameter_bounds(self) -> List[tuple]:
        """
        Retorna os limites (min, max) para cada parâmetro.
        
        Returns:
            Lista de tuplas (min, max) para cada parâmetro.
        """
        pass
    
    @abstractmethod
    def parameter_names(self) -> List[str]:
        """
        Retorna os nomes dos parâmetros.
        
        Returns:
            Lista com nomes dos parâmetros.
        """
        pass


class BlackScholesModel(PricingModel):
    """
    Modelo de Black-Scholes para precificação de opções europeias.
    
    O único parâmetro a calibrar é a volatilidade implícita (σ).
    
    Fórmula:
    - Call: C = S*N(d1) - K*e^(-r*T)*N(d2)
    - Put:  P = K*e^(-r*T)*N(-d2) - S*N(-d1)
    
    onde:
    - d1 = [ln(S/K) + (r + σ²/2)*T] / (σ*√T)
    - d2 = d1 - σ*√T
    """
    
    def price(self, option: OptionData, parameters: np.ndarray) -> float:
        """
        Calcula o preço usando Black-Scholes.
        
        Args:
            option: Dados da opção.
            parameters: Array com [volatility].
            
        Returns:
            Preço da opção.
        """
        if len(parameters) != 1:
            raise ValueError("Black-Scholes requer exatamente 1 parâmetro (volatilidade)")
        
        sigma = parameters[0]
        
        # Extrai dados da opção
        S = option.spot_price
        K = option.strike_price
        T = option.time_to_maturity
        r = option.risk_free_rate
        
        # Tratamento de casos especiais
        if sigma <= 0:
            # Volatilidade inválida, retorna valor intrínseco
            if option.option_type == 'call':
                return max(S - K * np.exp(-r * T), 0)
            else:
                return max(K * np.exp(-r * T) - S, 0)
        
        if T <= 0:
            # Opção expirada
            if option.option_type == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        # Calcula d1 e d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calcula o preço baseado no tipo
        if option.option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def parameter_bounds(self) -> List[tuple]:
        """
        Limites para volatilidade.
        
        Returns:
            [(sigma_min, sigma_max)]
        """
        return [(0.01, 2.0)]  # Volatilidade entre 1% e 200%
    
    def parameter_names(self) -> List[str]:
        """
        Nomes dos parâmetros.
        
        Returns:
            ['volatility']
        """
        return ['volatility']
    
    def implied_volatility_analytical(
        self, 
        option: OptionData, 
        tolerance: float = 1e-6,
        max_iterations: int = 100
    ) -> float:
        """
        Calcula a volatilidade implícita usando Newton-Raphson.
        
        Args:
            option: Dados da opção com market_price.
            tolerance: Tolerância para convergência.
            max_iterations: Máximo de iterações.
            
        Returns:
            Volatilidade implícita.
        """
        # Estimativa inicial usando aproximação de Brenner-Subrahmanyam
        S = option.spot_price
        K = option.strike_price
        T = option.time_to_maturity
        r = option.risk_free_rate
        C = option.market_price
        
        # Estimativa inicial
        sigma = np.sqrt(2 * np.pi / T) * (C / S)
        sigma = max(0.01, min(sigma, 2.0))  # Limita ao range válido
        
        for _ in range(max_iterations):
            # Calcula preço e vega
            price = self.price(option, np.array([sigma]))
            vega = self._calculate_vega(option, sigma)
            
            # Verifica convergência
            diff = C - price
            if abs(diff) < tolerance:
                return sigma
            
            # Atualiza usando Newton-Raphson
            if vega > 1e-10:  # Evita divisão por zero
                sigma = sigma + diff / vega
                sigma = max(0.01, min(sigma, 2.0))  # Mantém no range
            else:
                break
        
        return sigma
    
    def _calculate_vega(self, option: OptionData, sigma: float) -> float:
        """
        Calcula vega (derivada do preço em relação à volatilidade).
        
        Args:
            option: Dados da opção.
            sigma: Volatilidade.
            
        Returns:
            Vega da opção.
        """
        S = option.spot_price
        K = option.strike_price
        T = option.time_to_maturity
        r = option.risk_free_rate
        
        if sigma <= 0 or T <= 0:
            return 0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return vega


class HestonModel(PricingModel):
    """
    Modelo de Heston para volatilidade estocástica.
    
    Parâmetros a calibrar:
    - v0: Volatilidade inicial
    - kappa: Taxa de reversão à média
    - theta: Volatilidade de longo prazo
    - sigma: Volatilidade da volatilidade
    - rho: Correlação entre os processos de Wiener
    
    Nota: Implementação simplificada usando aproximação de Monte Carlo.
    Para produção, usar método de integração de Carr-Madan ou FFT.
    """
    
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        """
        Inicializa o modelo de Heston.
        
        Args:
            n_simulations: Número de simulações Monte Carlo.
            random_seed: Seed para reprodutibilidade.
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
    
    def price(self, option: OptionData, parameters: np.ndarray) -> float:
        """
        Calcula o preço usando simulação de Monte Carlo.
        
        Args:
            option: Dados da opção.
            parameters: Array com [v0, kappa, theta, sigma, rho].
            
        Returns:
            Preço da opção.
        """
        if len(parameters) != 5:
            raise ValueError("Heston requer 5 parâmetros: [v0, kappa, theta, sigma, rho]")
        
        v0, kappa, theta, sigma_v, rho = parameters
        
        # Validação da condição de Feller
        if 2 * kappa * theta < sigma_v**2:
            print("Aviso: Condição de Feller violada (2*kappa*theta >= sigma_v^2)")
        
        # Extrai dados
        S0 = option.spot_price
        K = option.strike_price
        T = option.time_to_maturity
        r = option.risk_free_rate
        
        # Simulação Monte Carlo
        np.random.seed(self.random_seed)
        dt = T / 100  # 100 passos de tempo
        n_steps = int(T / dt)
        
        # Inicializa arrays
        S = np.full(self.n_simulations, S0)
        v = np.full(self.n_simulations, v0)
        
        # Simula caminhos
        for _ in range(n_steps):
            # Gera choques correlacionados
            z1 = np.random.standard_normal(self.n_simulations)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(self.n_simulations)
            
            # Atualiza preço do ativo
            S = S * np.exp((r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0) * dt) * z1)
            
            # Atualiza volatilidade (esquema de Euler com truncamento)
            v = v + kappa * (theta - v) * dt + sigma_v * np.sqrt(np.maximum(v, 0) * dt) * z2
            v = np.maximum(v, 0)  # Garante não-negatividade
        
        # Calcula payoff
        if option.option_type == 'call':
            payoff = np.maximum(S - K, 0)
        else:
            payoff = np.maximum(K - S, 0)
        
        # Desconta e tira média
        price = np.exp(-r * T) * np.mean(payoff)
        
        return price
    
    def parameter_bounds(self) -> List[tuple]:
        """
        Limites para parâmetros de Heston.
        
        Returns:
            [(v0_min, v0_max), (kappa_min, kappa_max), ...]
        """
        return [
            (0.01, 1.0),   # v0: volatilidade inicial
            (0.1, 10.0),   # kappa: taxa de reversão
            (0.01, 1.0),   # theta: volatilidade de longo prazo
            (0.01, 2.0),   # sigma: vol da vol
            (-0.99, 0.99)  # rho: correlação
        ]
    
    def parameter_names(self) -> List[str]:
        """Nomes dos parâmetros."""
        return ['v0', 'kappa', 'theta', 'sigma_v', 'rho']


def calculate_mse(
    options: List[OptionData], 
    model: PricingModel, 
    parameters: np.ndarray
) -> float:
    """
    Calcula o erro quadrático médio entre preços de mercado e modelo.
    
    Args:
        options: Lista de opções.
        model: Modelo de precificação.
        parameters: Parâmetros do modelo.
        
    Returns:
        MSE (Mean Squared Error).
    """
    errors = []
    
    for option in options:
        theoretical_price = model.price(option, parameters)
        market_price = option.market_price
        errors.append((theoretical_price - market_price) ** 2)
    
    return np.mean(errors)


def calculate_rmse(
    options: List[OptionData], 
    model: PricingModel, 
    parameters: np.ndarray
) -> float:
    """
    Calcula o RMSE (Root Mean Squared Error).
    
    Args:
        options: Lista de opções.
        model: Modelo de precificação.
        parameters: Parâmetros do modelo.
        
    Returns:
        RMSE.
    """
    return np.sqrt(calculate_mse(options, model, parameters))
