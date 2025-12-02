"""
Script principal para calibração de modelos de volatilidade.

Demonstra o uso completo do sistema de calibração usando algoritmos evolutivos.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Adiciona o diretório src ao path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from src.data_loader import RiskFreeRateLoader
from src.synthetic_data import SyntheticOptionGenerator
from src.pricing_models import BlackScholesModel, HestonModel, calculate_rmse
from src.evolutionary_algo import EvolutionaryAlgorithm, EvolutionaryConfig


def print_header(title: str) -> None:
    """Imprime um cabeçalho formatado."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_section(title: str) -> None:
    """Imprime um cabeçalho de seção."""
    print(f"\n{title}")
    print("-" * len(title))


def calibrate_black_scholes(
    csv_path: str,
    n_options: int = 30,
    true_volatility: float = 0.25,
    population_size: int = 100,
    n_generations: int = 50,
    plot: bool = True
) -> dict:
    """
    Executa calibração do modelo Black-Scholes.
    
    Args:
        csv_path: Caminho para o CSV de taxas de juros.
        n_options: Número de opções sintéticas a gerar.
        true_volatility: Volatilidade verdadeira para dados sintéticos.
        population_size: Tamanho da população do algoritmo evolutivo.
        n_generations: Número de gerações.
        plot: Se True, gera gráficos de convergência.
        
    Returns:
        Dicionário com resultados da calibração.
    """
    print_header("CALIBRAÇÃO BLACK-SCHOLES COM ALGORITMO EVOLUTIVO")
    
    # 1. Carrega dados de taxa de juros
    print_section("1. Carregando dados de taxa de juros")
    rate_loader = RiskFreeRateLoader(csv_path)
    print(f"✓ {rate_loader}")
    
    # 2. Gera dados sintéticos
    print_section("2. Gerando opções sintéticas")
    generator = SyntheticOptionGenerator(rate_loader)
    options, true_vol = generator.generate_options(
        n_options=n_options,
        true_volatility=true_volatility,
        add_noise=True,
        noise_std=0.3
    )
    print(f"✓ Geradas {len(options)} opções")
    print(f"✓ Volatilidade verdadeira: {true_vol:.4f} ({true_vol*100:.2f}%)")
    
    # Mostra amostra das opções
    print("\nAmostra de opções geradas:")
    for i, opt in enumerate(options[:5]):
        print(f"  [{i+1}] {opt.option_type.upper():4s} | "
              f"S={opt.spot_price:6.2f} | K={opt.strike_price:6.2f} | "
              f"T={opt.time_to_maturity:.2f}y | r={opt.risk_free_rate*100:5.2f}% | "
              f"Preço Mercado={opt.market_price:6.3f}")
    if len(options) > 5:
        print(f"  ... e mais {len(options)-5} opções")
    
    # 3. Configura modelo
    print_section("3. Configurando modelo Black-Scholes")
    model = BlackScholesModel()
    print(f"✓ Parâmetros a calibrar: {model.parameter_names()}")
    print(f"✓ Limites: {model.parameter_bounds()}")
    
    # 4. Configura algoritmo evolutivo
    print_section("4. Configurando Algoritmo Evolutivo")
    config = EvolutionaryConfig(
        population_size=population_size,
        n_generations=n_generations,
        crossover_rate=0.8,
        mutation_rate=0.15,
        mutation_std=0.1,
        tournament_size=5,
        elitism_count=2,
        random_seed=42
    )
    print(f"✓ População: {config.population_size}")
    print(f"✓ Gerações: {config.n_generations}")
    print(f"✓ Taxa de Crossover: {config.crossover_rate}")
    print(f"✓ Taxa de Mutação: {config.mutation_rate}")
    
    # 5. Executa calibração
    print_section("5. Executando Calibração")
    ea = EvolutionaryAlgorithm(model, options, config)
    results = ea.evolve(verbose=True)
    
    # 6. Resultados
    print_section("6. Resultados da Calibração")
    calibrated_vol = results['best_parameters'][0]
    print(f"\n{'Parâmetro':<20} {'Verdadeiro':>12} {'Calibrado':>12} {'Erro (%)':>12}")
    print("-" * 60)
    print(f"{'Volatilidade':<20} {true_vol:12.6f} {calibrated_vol:12.6f} "
          f"{abs(calibrated_vol - true_vol) / true_vol * 100:11.2f}%")
    
    print(f"\nMétricas de Erro:")
    print(f"  MSE:  {results['best_mse']:.6f}")
    print(f"  RMSE: {results['best_rmse']:.6f}")
    print(f"  Fitness: {results['best_fitness']:.6f}")
    
    # 7. Validação: compara preços
    print_section("7. Validação dos Preços")
    print(f"\n{'#':<4} {'Tipo':>6} {'Preço Mercado':>14} {'Preço Modelo':>14} {'Erro':>10}")
    print("-" * 55)
    
    errors = []
    for i, opt in enumerate(options[:10]):
        model_price = model.price(opt, results['best_parameters'])
        error = model_price - opt.market_price
        errors.append(error)
        print(f"{i+1:<4} {opt.option_type.upper():>6} {opt.market_price:14.4f} "
              f"{model_price:14.4f} {error:10.4f}")
    
    if len(options) > 10:
        print(f"... e mais {len(options)-10} opções")
    
    # 8. Gráfico de convergência
    if plot:
        print_section("8. Gerando Gráficos")
        plt.figure(figsize=(14, 5))
        
        # Subplot 1: Convergência do Fitness
        plt.subplot(1, 2, 1)
        plt.plot(results['fitness_history'], label='Melhor Fitness', linewidth=2)
        plt.plot(results['avg_fitness_history'], label='Fitness Médio', 
                 linewidth=2, alpha=0.7)
        plt.xlabel('Geração')
        plt.ylabel('Fitness')
        plt.title('Convergência do Algoritmo Evolutivo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: MSE ao longo das gerações
        plt.subplot(1, 2, 2)
        mse_history = [1.0/f - 1.0 for f in results['fitness_history']]
        plt.plot(mse_history, color='red', linewidth=2)
        plt.xlabel('Geração')
        plt.ylabel('MSE')
        plt.title('Evolução do Erro (MSE)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.tight_layout()
        
        # Salva gráfico
        output_path = Path(__file__).parent / 'calibration_convergence.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Gráfico salvo em: {output_path}")
        
        # plt.show()  # Descomente para mostrar interativamente
    
    return results


def calibrate_heston(
    csv_path: str,
    n_options: int = 20,
    population_size: int = 150,
    n_generations: int = 100
) -> dict:
    """
    Exemplo de calibração do modelo Heston (mais computacionalmente intenso).
    
    Args:
        csv_path: Caminho para o CSV.
        n_options: Número de opções.
        population_size: Tamanho da população.
        n_generations: Número de gerações.
        
    Returns:
        Resultados da calibração.
    """
    print_header("CALIBRAÇÃO HESTON (EXEMPLO AVANÇADO)")
    
    print("\n⚠️  AVISO: Calibração de Heston é computacionalmente intensiva!")
    print("    Este exemplo usa Monte Carlo e pode levar vários minutos.")
    
    # Carrega dados
    print_section("1. Preparando dados")
    rate_loader = RiskFreeRateLoader(csv_path)
    generator = SyntheticOptionGenerator(rate_loader)
    
    # Gera opções sintéticas (menos para economizar tempo)
    options, _ = generator.generate_options(
        n_options=n_options,
        true_volatility=0.25,
        add_noise=True
    )
    print(f"✓ {len(options)} opções geradas")
    
    # Configura modelo Heston
    print_section("2. Configurando Modelo Heston")
    model = HestonModel(n_simulations=5000)  # Reduzido para velocidade
    print(f"✓ Parâmetros: {model.parameter_names()}")
    print(f"✓ Limites: {model.parameter_bounds()}")
    
    # Configura EA com mais gerações
    print_section("3. Configurando EA (configuração intensiva)")
    config = EvolutionaryConfig(
        population_size=population_size,
        n_generations=n_generations,
        crossover_rate=0.8,
        mutation_rate=0.2,
        tournament_size=5,
        elitism_count=3
    )
    
    # Calibra
    print_section("4. Executando Calibração (isso pode demorar...)")
    ea = EvolutionaryAlgorithm(model, options, config)
    results = ea.evolve(verbose=True)
    
    # Resultados
    print_section("5. Resultados")
    params = ea.get_parameter_dict()
    for name, value in params.items():
        print(f"  {name:12s}: {value:.6f}")
    
    print(f"\n  MSE:  {results['best_mse']:.6f}")
    print(f"  RMSE: {results['best_rmse']:.6f}")
    
    return results


def demonstrate_portfolio_calibration(csv_path: str) -> None:
    """
    Demonstra calibração usando um portfólio realista de opções.
    
    Args:
        csv_path: Caminho para o CSV.
    """
    print_header("CALIBRAÇÃO COM PORTFÓLIO REALISTA")
    
    # Carrega dados
    rate_loader = RiskFreeRateLoader(csv_path)
    generator = SyntheticOptionGenerator(rate_loader)
    
    # Gera portfólio
    print_section("Gerando Portfólio de Opções")
    options, true_vol = generator.generate_portfolio(
        spot_price=100.0,
        true_volatility=0.22
    )
    print(f"✓ Portfólio com {len(options)} opções")
    print(f"✓ Volatilidade verdadeira: {true_vol:.4f}")
    
    # Mostra estrutura do portfólio
    print("\nEstrutura do Portfólio:")
    strikes = sorted(set(opt.strike_price for opt in options))
    maturities = sorted(set(opt.time_to_maturity for opt in options))
    print(f"  Strikes: {[f'{s:.1f}' for s in strikes]}")
    print(f"  Maturidades: {[f'{m:.2f}y' for m in maturities]}")
    
    # Calibra
    print_section("Calibrando...")
    model = BlackScholesModel()
    config = EvolutionaryConfig(
        population_size=80,
        n_generations=40,
        random_seed=42
    )
    
    ea = EvolutionaryAlgorithm(model, options, config)
    results = ea.evolve(verbose=False)
    
    calibrated_vol = results['best_parameters'][0]
    print(f"\n✓ Volatilidade Calibrada: {calibrated_vol:.6f}")
    print(f"✓ Volatilidade Verdadeira: {true_vol:.6f}")
    print(f"✓ Erro: {abs(calibrated_vol - true_vol):.6f} "
          f"({abs(calibrated_vol - true_vol) / true_vol * 100:.2f}%)")


def main():
    """Função principal."""
    # Caminho para o CSV
    csv_path = Path(__file__).parent / 'databasePy.csv'
    
    if not csv_path.exists():
        print(f"❌ Erro: Arquivo {csv_path} não encontrado!")
        print("   Certifique-se de que databasePy.csv está no diretório do projeto.")
        return
    
    print("\n" + "🎯" * 40)
    print(" SISTEMA DE CALIBRAÇÃO DE MODELOS DE VOLATILIDADE")
    print(" Usando Algoritmos Evolutivos")
    print("🎯" * 40)
    
    # Menu de opções
    print("\nEscolha o modo de execução:")
    print("  [1] Calibração Black-Scholes (Rápido - Recomendado)")
    print("  [2] Calibração com Portfólio Realista")
    print("  [3] Calibração Heston (Lento - Avançado)")
    print("  [4] Executar Todos")
    
    choice = input("\nOpção [1]: ").strip() or "1"
    
    try:
        if choice == "1":
            calibrate_black_scholes(str(csv_path))
        elif choice == "2":
            demonstrate_portfolio_calibration(str(csv_path))
        elif choice == "3":
            calibrate_heston(str(csv_path))
        elif choice == "4":
            calibrate_black_scholes(str(csv_path))
            demonstrate_portfolio_calibration(str(csv_path))
            
            print("\n⚠️  Calibração Heston foi pulada (muito lento).")
            print("    Para executar, escolha opção [3] separadamente.")
        else:
            print("❌ Opção inválida!")
            return
        
        print_header("CALIBRAÇÃO CONCLUÍDA COM SUCESSO!")
        print("\n✅ Todos os processos foram executados.")
        print("📊 Verifique os resultados e gráficos gerados.")
        
    except Exception as e:
        print(f"\n❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
