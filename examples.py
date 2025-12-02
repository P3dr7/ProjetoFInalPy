"""
Script auxiliar para exemplos rápidos e experimentação.

Use este script para testar funcionalidades específicas rapidamente.
"""

import sys
from pathlib import Path

# Adiciona src ao path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.data_loader import RiskFreeRateLoader
from src.pricing_models import BlackScholesModel, OptionData
from datetime import datetime
import numpy as np


def example_load_rates():
    """Exemplo: Carrega e consulta taxas de juros."""
    print("=" * 60)
    print("EXEMPLO 1: Carregamento de Taxas de Juros")
    print("=" * 60)
    
    loader = RiskFreeRateLoader('databasePy.csv')
    print(f"\n{loader}")
    
    # Consulta taxa
    date = datetime(2022, 12, 30)
    rate_90d = loader.get_risk_free_rate(date, 90)
    rate_180d = loader.get_risk_free_rate(date, 180)
    
    print(f"\nTaxas em {date.date()}:")
    print(f"  90 dias:  {rate_90d:.4f} ({rate_90d*100:.2f}%)")
    print(f"  180 dias: {rate_180d:.4f} ({rate_180d*100:.2f}%)")


def example_black_scholes():
    """Exemplo: Precificação Black-Scholes."""
    print("\n" + "=" * 60)
    print("EXEMPLO 2: Precificação Black-Scholes")
    print("=" * 60)
    
    model = BlackScholesModel()
    
    # Call ATM
    call = OptionData(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=0.25,  # 3 meses
        risk_free_rate=0.05,
        option_type='call',
        market_price=0.0
    )
    
    volatility = 0.25
    call_price = model.price(call, np.array([volatility]))
    
    print(f"\nCall Option ATM:")
    print(f"  Spot: ${call.spot_price}")
    print(f"  Strike: ${call.strike_price}")
    print(f"  Maturity: {call.time_to_maturity * 365:.0f} days")
    print(f"  Volatility: {volatility*100:.1f}%")
    print(f"  → Preço: ${call_price:.4f}")
    
    # Put ATM
    put = OptionData(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=0.25,
        risk_free_rate=0.05,
        option_type='put',
        market_price=0.0
    )
    
    put_price = model.price(put, np.array([volatility]))
    
    print(f"\nPut Option ATM:")
    print(f"  → Preço: ${put_price:.4f}")
    
    # Verifica put-call parity
    S = call.spot_price
    K = call.strike_price
    r = call.risk_free_rate
    T = call.time_to_maturity
    
    parity_lhs = call_price - put_price
    parity_rhs = S - K * np.exp(-r * T)
    
    print(f"\nPut-Call Parity:")
    print(f"  C - P = {parity_lhs:.4f}")
    print(f"  S - K*e^(-rT) = {parity_rhs:.4f}")
    print(f"  ✓ Diferença: {abs(parity_lhs - parity_rhs):.6f}")


def example_synthetic_data():
    """Exemplo: Geração de dados sintéticos."""
    print("\n" + "=" * 60)
    print("EXEMPLO 3: Dados Sintéticos")
    print("=" * 60)
    
    from src.synthetic_data import SyntheticOptionGenerator
    
    loader = RiskFreeRateLoader('databasePy.csv')
    generator = SyntheticOptionGenerator(loader)
    
    options, true_vol = generator.generate_options(
        n_options=5,
        true_volatility=0.30,
        add_noise=True
    )
    
    print(f"\nGeradas {len(options)} opções com σ = {true_vol:.2f}")
    print(f"\n{'#':<3} {'Tipo':>6} {'Strike':>8} {'Dias':>6} {'Preço':>10}")
    print("-" * 40)
    
    for i, opt in enumerate(options):
        print(f"{i+1:<3} {opt.option_type.upper():>6} "
              f"{opt.strike_price:8.2f} "
              f"{opt.time_to_maturity*365:6.0f} "
              f"{opt.market_price:10.4f}")


def example_mini_calibration():
    """Exemplo: Mini calibração."""
    print("\n" + "=" * 60)
    print("EXEMPLO 4: Mini Calibração")
    print("=" * 60)
    
    from src.synthetic_data import SyntheticOptionGenerator
    from src.evolutionary_algo import EvolutionaryAlgorithm, EvolutionaryConfig
    
    # Gera dados
    loader = RiskFreeRateLoader('databasePy.csv')
    generator = SyntheticOptionGenerator(loader)
    
    true_vol = 0.28
    options, _ = generator.generate_options(
        n_options=10,
        true_volatility=true_vol,
        add_noise=True,
        noise_std=0.2
    )
    
    print(f"\nVolatilidade verdadeira: {true_vol:.4f}")
    
    # Configura EA
    model = BlackScholesModel()
    config = EvolutionaryConfig(
        population_size=30,
        n_generations=20,
        random_seed=42
    )
    
    print(f"Calibrando com {config.population_size} indivíduos, "
          f"{config.n_generations} gerações...")
    
    # Calibra
    ea = EvolutionaryAlgorithm(model, options, config)
    results = ea.evolve(verbose=False)
    
    calibrated_vol = results['best_parameters'][0]
    error = abs(calibrated_vol - true_vol)
    
    print(f"\n✓ Volatilidade calibrada: {calibrated_vol:.4f}")
    print(f"✓ Erro absoluto: {error:.4f}")
    print(f"✓ Erro relativo: {error/true_vol*100:.2f}%")
    print(f"✓ RMSE: {results['best_rmse']:.6f}")


if __name__ == "__main__":
    print("\n🎯 EXEMPLOS DE USO DO SISTEMA\n")
    
    try:
        example_load_rates()
        example_black_scholes()
        example_synthetic_data()
        example_mini_calibration()
        
        print("\n" + "=" * 60)
        print("✅ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
