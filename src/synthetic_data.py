"""
Módulo para geração de dados sintéticos de opções.

Gera opções sintéticas com preços de mercado para testar a calibração.
"""

from typing import List, Optional
from datetime import datetime, timedelta
import numpy as np
from .pricing_models import OptionData, BlackScholesModel
from .data_loader import RiskFreeRateLoader


class SyntheticOptionGenerator:
    """Gerador de dados sintéticos de opções para testes de calibração."""
    
    def __init__(
        self, 
        rate_loader: RiskFreeRateLoader,
        random_seed: int = 42
    ):
        """
        Inicializa o gerador de dados sintéticos.
        
        Args:
            rate_loader: Carregador de taxas de juros.
            random_seed: Seed para reprodutibilidade.
        """
        self.rate_loader = rate_loader
        self.random_seed = random_seed
        self.bs_model = BlackScholesModel()
        np.random.seed(random_seed)
    
    def generate_options(
        self,
        n_options: int = 20,
        spot_price: float = 100.0,
        true_volatility: float = 0.25,
        date: Optional[datetime] = None,
        add_noise: bool = True,
        noise_std: float = 0.5
    ) -> tuple[List[OptionData], float]:
        """
        Gera um conjunto de opções sintéticas com preços de mercado.
        
        Args:
            n_options: Número de opções a gerar.
            spot_price: Preço atual do ativo subjacente.
            true_volatility: Volatilidade "verdadeira" usada para gerar preços.
            date: Data de referência (padrão: data mais recente no dataset).
            add_noise: Se True, adiciona ruído aos preços de mercado.
            noise_std: Desvio padrão do ruído (em % do preço).
            
        Returns:
            Tupla (lista de opções, volatilidade verdadeira).
        """
        if date is None:
            # Usa a data mais recente disponível
            _, max_date = self.rate_loader.get_available_date_range()
            date = max_date
        
        options = []
        
        # Gera variedade de opções
        for i in range(n_options):
            # Randomiza parâmetros para criar diversidade
            
            # Strike: varia de 80% a 120% do spot (ATM, ITM, OTM)
            moneyness = np.random.uniform(0.8, 1.2)
            strike = spot_price * moneyness
            
            # Maturidade: entre 30 e 365 dias
            maturity_days = np.random.randint(30, 366)
            time_to_maturity = maturity_days / 365.0
            
            # Tipo: 50% calls, 50% puts
            option_type = 'call' if i % 2 == 0 else 'put'
            
            # Obtém taxa livre de risco
            risk_free_rate = self.rate_loader.get_risk_free_rate(
                date, 
                maturity_days,
                interpolate_date=True,
                interpolate_maturity=True
            )
            
            # Cria objeto OptionData temporário (sem market_price)
            temp_option = OptionData(
                spot_price=spot_price,
                strike_price=strike,
                time_to_maturity=time_to_maturity,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                market_price=0.0  # Será calculado abaixo
            )
            
            # Calcula preço teórico com volatilidade verdadeira
            theoretical_price = self.bs_model.price(
                temp_option, 
                np.array([true_volatility])
            )
            
            # Adiciona ruído para simular imperfeições de mercado
            if add_noise:
                noise = np.random.normal(0, noise_std)
                market_price = max(0.01, theoretical_price + noise)
            else:
                market_price = theoretical_price
            
            # Cria opção final com market_price
            option = OptionData(
                spot_price=spot_price,
                strike_price=strike,
                time_to_maturity=time_to_maturity,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                market_price=market_price
            )
            
            options.append(option)
        
        return options, true_volatility
    
    def generate_portfolio(
        self,
        spot_price: float = 100.0,
        true_volatility: float = 0.25,
        date: Optional[datetime] = None
    ) -> tuple[List[OptionData], float]:
        """
        Gera um portfólio realista de opções.
        
        Cria opções com diferentes strikes e maturidades para simular
        um cenário real de calibração.
        
        Args:
            spot_price: Preço do ativo subjacente.
            true_volatility: Volatilidade verdadeira.
            date: Data de referência.
            
        Returns:
            Tupla (lista de opções, volatilidade verdadeira).
        """
        if date is None:
            _, max_date = self.rate_loader.get_available_date_range()
            date = max_date
        
        options = []
        
        # Define grid de strikes (90%, 95%, 100%, 105%, 110% do spot)
        strike_ratios = [0.90, 0.95, 1.00, 1.05, 1.10]
        
        # Define maturidades (1, 3, 6 meses)
        maturities_days = [30, 90, 180]
        
        for strike_ratio in strike_ratios:
            for maturity_days in maturities_days:
                strike = spot_price * strike_ratio
                time_to_maturity = maturity_days / 365.0
                
                # Obtém taxa livre de risco
                risk_free_rate = self.rate_loader.get_risk_free_rate(
                    date,
                    maturity_days,
                    interpolate_date=True,
                    interpolate_maturity=True
                )
                
                # Gera call e put para cada combinação
                for option_type in ['call', 'put']:
                    temp_option = OptionData(
                        spot_price=spot_price,
                        strike_price=strike,
                        time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate,
                        option_type=option_type,
                        market_price=0.0
                    )
                    
                    # Calcula preço teórico
                    theoretical_price = self.bs_model.price(
                        temp_option,
                        np.array([true_volatility])
                    )
                    
                    # Adiciona ruído proporcional ao preço
                    noise_pct = np.random.normal(0, 0.02)  # ±2% noise
                    market_price = max(0.01, theoretical_price * (1 + noise_pct))
                    
                    # Cria opção final
                    option = OptionData(
                        spot_price=spot_price,
                        strike_price=strike,
                        time_to_maturity=time_to_maturity,
                        risk_free_rate=risk_free_rate,
                        option_type=option_type,
                        market_price=market_price
                    )
                    
                    options.append(option)
        
        return options, true_volatility
    
    def generate_smile_scenario(
        self,
        spot_price: float = 100.0,
        maturity_days: int = 90,
        date: Optional[datetime] = None
    ) -> List[OptionData]:
        """
        Gera opções para ilustrar volatility smile/skew.
        
        Args:
            spot_price: Preço do ativo.
            maturity_days: Maturidade das opções.
            date: Data de referência.
            
        Returns:
            Lista de opções com diferentes strikes.
        """
        if date is None:
            _, max_date = self.rate_loader.get_available_date_range()
            date = max_date
        
        options = []
        time_to_maturity = maturity_days / 365.0
        
        # Obtém taxa livre de risco
        risk_free_rate = self.rate_loader.get_risk_free_rate(
            date,
            maturity_days,
            interpolate_date=True,
            interpolate_maturity=True
        )
        
        # Grid de strikes para capturar o smile
        strike_ratios = np.linspace(0.80, 1.20, 11)
        
        for strike_ratio in strike_ratios:
            strike = spot_price * strike_ratio
            
            # Simula volatility smile (maior volatilidade OTM)
            # Modelo simples de smile: volatilidade aumenta longe do ATM
            atm_vol = 0.25
            smile_effect = 0.05 * ((strike_ratio - 1.0) ** 2)
            implied_vol = atm_vol + smile_effect
            
            for option_type in ['call', 'put']:
                temp_option = OptionData(
                    spot_price=spot_price,
                    strike_price=strike,
                    time_to_maturity=time_to_maturity,
                    risk_free_rate=risk_free_rate,
                    option_type=option_type,
                    market_price=0.0
                )
                
                # Preço com volatilidade específica do smile
                theoretical_price = self.bs_model.price(
                    temp_option,
                    np.array([implied_vol])
                )
                
                option = OptionData(
                    spot_price=spot_price,
                    strike_price=strike,
                    time_to_maturity=time_to_maturity,
                    risk_free_rate=risk_free_rate,
                    option_type=option_type,
                    market_price=theoretical_price
                )
                
                options.append(option)
        
        return options
