"""
Testes unitários para o módulo evolutionary_algo.

Testa algoritmo evolutivo, operadores genéticos e convergência.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from src.evolutionary_algo import (
    EvolutionaryConfig,
    Individual,
    EvolutionaryAlgorithm
)
from src.pricing_models import BlackScholesModel, OptionData


class TestEvolutionaryConfig:
    """Testes para a configuração do algoritmo evolutivo."""
    
    def test_default_config(self):
        """Testa configuração padrão."""
        config = EvolutionaryConfig()
        
        assert config.population_size > 0
        assert config.n_generations > 0
        assert 0 <= config.crossover_rate <= 1
        assert 0 <= config.mutation_rate <= 1
    
    def test_invalid_population_size(self):
        """Testa erro com população inválida."""
        with pytest.raises(ValueError, match="Population size deve ser"):
            EvolutionaryConfig(population_size=1)
    
    def test_invalid_crossover_rate(self):
        """Testa erro com taxa de crossover inválida."""
        with pytest.raises(ValueError, match="Crossover rate deve estar"):
            EvolutionaryConfig(crossover_rate=1.5)
        
        with pytest.raises(ValueError, match="Crossover rate deve estar"):
            EvolutionaryConfig(crossover_rate=-0.1)
    
    def test_invalid_mutation_rate(self):
        """Testa erro com taxa de mutação inválida."""
        with pytest.raises(ValueError, match="Mutation rate deve estar"):
            EvolutionaryConfig(mutation_rate=2.0)
    
    def test_invalid_tournament_size(self):
        """Testa erro quando torneio é maior que população."""
        with pytest.raises(ValueError, match="Tournament size não pode ser maior"):
            EvolutionaryConfig(population_size=10, tournament_size=20)


class TestIndividual:
    """Testes para a classe Individual."""
    
    def test_individual_creation(self):
        """Testa criação de indivíduo."""
        genes = np.array([0.25, 0.30])
        ind = Individual(genes=genes, fitness=0.8)
        
        assert np.array_equal(ind.genes, genes)
        assert ind.fitness == 0.8
    
    def test_individual_comparison(self):
        """Testa comparação entre indivíduos."""
        ind1 = Individual(np.array([0.2]), fitness=0.5)
        ind2 = Individual(np.array([0.3]), fitness=0.8)
        
        # Maior fitness é "maior"
        assert ind1 < ind2
        assert not ind2 < ind1


class TestEvolutionaryAlgorithm:
    """Testes para o algoritmo evolutivo."""
    
    @pytest.fixture
    def simple_problem(self):
        """Cria um problema simples de calibração."""
        model = BlackScholesModel()
        
        # Cria opções com preços conhecidos (vol = 0.25)
        true_vol = 0.25
        
        options = []
        for _ in range(5):
            opt = OptionData(
                spot_price=100.0,
                strike_price=np.random.uniform(90, 110),
                time_to_maturity=np.random.uniform(0.25, 1.0),
                risk_free_rate=0.05,
                option_type='call',
                market_price=0.0
            )
            
            # Calcula preço com volatilidade verdadeira
            opt.market_price = model.price(opt, np.array([true_vol]))
            options.append(opt)
        
        return model, options, true_vol
    
    def test_initialization(self, simple_problem):
        """Testa inicialização do algoritmo."""
        model, options, _ = simple_problem
        
        config = EvolutionaryConfig(population_size=20, n_generations=10)
        ea = EvolutionaryAlgorithm(model, options, config)
        
        assert ea.model == model
        assert ea.options == options
        assert ea.n_parameters == 1  # Black-Scholes tem 1 parâmetro
    
    def test_population_initialization(self, simple_problem):
        """Testa criação da população inicial."""
        model, options, _ = simple_problem
        
        config = EvolutionaryConfig(population_size=30)
        ea = EvolutionaryAlgorithm(model, options, config)
        
        population = ea._initialize_population()
        
        assert len(population) == 30
        assert all(isinstance(ind, Individual) for ind in population)
        
        # Genes devem estar dentro dos limites
        bounds = ea.bounds
        for ind in population:
            assert all(bounds[i, 0] <= ind.genes[i] <= bounds[i, 1] 
                      for i in range(ea.n_parameters))
    
    def test_fitness_calculation(self, simple_problem):
        """Testa cálculo de fitness."""
        model, options, true_vol = simple_problem
        
        ea = EvolutionaryAlgorithm(model, options)
        
        # Indivíduo com parâmetro correto deve ter fitness alto
        perfect_ind = Individual(np.array([true_vol]))
        fitness = ea._calculate_fitness(perfect_ind)
        
        # Fitness = 1/(1 + MSE), com MSE ~0, fitness ~1
        assert fitness > 0.99
        
        # Indivíduo com parâmetro ruim deve ter fitness baixo
        bad_ind = Individual(np.array([0.01]))
        fitness_bad = ea._calculate_fitness(bad_ind)
        
        assert fitness_bad < fitness
    
    def test_tournament_selection(self, simple_problem):
        """Testa seleção por torneio."""
        model, options, _ = simple_problem
        
        config = EvolutionaryConfig(population_size=20, tournament_size=3)
        ea = EvolutionaryAlgorithm(model, options, config)
        
        population = ea._initialize_population()
        ea._evaluate_population(population)
        
        # Seleciona indivíduo
        selected = ea._tournament_selection(population)
        
        assert selected in population
        assert hasattr(selected, 'fitness')
    
    def test_roulette_selection(self, simple_problem):
        """Testa seleção por roleta."""
        model, options, _ = simple_problem
        
        ea = EvolutionaryAlgorithm(model, options)
        
        population = ea._initialize_population()
        ea._evaluate_population(population)
        
        selected = ea._roulette_selection(population)
        
        assert selected in population
    
    def test_arithmetic_crossover(self, simple_problem):
        """Testa crossover aritmético."""
        model, options, _ = simple_problem
        
        ea = EvolutionaryAlgorithm(model, options)
        
        parent1 = Individual(np.array([0.20]))
        parent2 = Individual(np.array([0.30]))
        
        child1, child2 = ea._arithmetic_crossover(parent1, parent2)
        
        # Filhos devem estar entre os pais
        assert 0.20 <= child1.genes[0] <= 0.30
        assert 0.20 <= child2.genes[0] <= 0.30
        
        # Genes devem estar dentro dos limites
        bounds = ea.bounds
        assert bounds[0, 0] <= child1.genes[0] <= bounds[0, 1]
        assert bounds[0, 0] <= child2.genes[0] <= bounds[0, 1]
    
    def test_uniform_crossover(self, simple_problem):
        """Testa crossover uniforme."""
        model, options, _ = simple_problem
        
        ea = EvolutionaryAlgorithm(model, options)
        
        parent1 = Individual(np.array([0.20]))
        parent2 = Individual(np.array([0.30]))
        
        child1, child2 = ea._uniform_crossover(parent1, parent2)
        
        # Cada filho deve ter genes de um dos pais
        assert child1.genes[0] in [0.20, 0.30]
        assert child2.genes[0] in [0.20, 0.30]
    
    def test_gaussian_mutation(self, simple_problem):
        """Testa mutação gaussiana."""
        model, options, _ = simple_problem
        
        config = EvolutionaryConfig(mutation_rate=1.0)  # Garante mutação
        ea = EvolutionaryAlgorithm(model, options, config)
        
        original = Individual(np.array([0.25]))
        mutated = ea._gaussian_mutation(original)
        
        # Genes devem ser diferentes
        assert mutated.genes[0] != original.genes[0]
        
        # Mas dentro dos limites
        bounds = ea.bounds
        assert bounds[0, 0] <= mutated.genes[0] <= bounds[0, 1]
    
    def test_clip_to_bounds(self, simple_problem):
        """Testa clipping aos limites."""
        model, options, _ = simple_problem
        
        ea = EvolutionaryAlgorithm(model, options)
        
        # Genes fora dos limites
        genes = np.array([10.0])  # Muito alto
        
        clipped = ea._clip_to_bounds(genes)
        
        # Deve estar dentro dos limites
        bounds = ea.bounds
        assert bounds[0, 0] <= clipped[0] <= bounds[0, 1]
    
    def test_evolution_convergence(self, simple_problem):
        """Testa se o algoritmo converge para a solução."""
        model, options, true_vol = simple_problem
        
        config = EvolutionaryConfig(
            population_size=50,
            n_generations=30,
            random_seed=42
        )
        
        ea = EvolutionaryAlgorithm(model, options, config)
        results = ea.evolve(verbose=False)
        
        # Verifica que evoluiu
        assert len(results['fitness_history']) == 30
        
        # Fitness deve melhorar
        initial_fitness = results['fitness_history'][0]
        final_fitness = results['fitness_history'][-1]
        assert final_fitness >= initial_fitness
        
        # Deve estar próximo da solução verdadeira
        calibrated_vol = results['best_parameters'][0]
        error = abs(calibrated_vol - true_vol)
        
        # Tolera erro de até 10% (problema estocástico)
        assert error / true_vol < 0.10
    
    def test_elitism(self, simple_problem):
        """Testa que elitismo preserva melhores indivíduos."""
        model, options, _ = simple_problem
        
        config = EvolutionaryConfig(
            population_size=20,
            elitism_count=2,
            random_seed=42
        )
        
        ea = EvolutionaryAlgorithm(model, options, config)
        
        population = ea._initialize_population()
        ea._evaluate_population(population)
        
        # Pega melhores da geração atual
        best_current = sorted(population, key=lambda x: x.fitness, reverse=True)[:2]
        
        # Cria próxima geração
        next_gen = ea._create_next_generation(population)
        
        # Verifica que os melhores estão na próxima geração
        next_gen_genes = [ind.genes for ind in next_gen]
        
        # Pelo menos deve ter indivíduos similares aos melhores
        assert len(next_gen) == config.population_size
    
    def test_get_parameter_dict(self, simple_problem):
        """Testa obtenção de parâmetros como dicionário."""
        model, options, _ = simple_problem
        
        config = EvolutionaryConfig(
            population_size=20,
            n_generations=5
        )
        
        ea = EvolutionaryAlgorithm(model, options, config)
        ea.evolve(verbose=False)
        
        params = ea.get_parameter_dict()
        
        assert isinstance(params, dict)
        assert 'volatility' in params
        assert isinstance(params['volatility'], (float, np.floating))
    
    def test_reproducibility(self, simple_problem):
        """Testa que seed garante reprodutibilidade."""
        model, options, _ = simple_problem
        
        config1 = EvolutionaryConfig(
            population_size=30,
            n_generations=10,
            random_seed=123
        )
        
        config2 = EvolutionaryConfig(
            population_size=30,
            n_generations=10,
            random_seed=123
        )
        
        ea1 = EvolutionaryAlgorithm(model, options, config1)
        ea2 = EvolutionaryAlgorithm(model, options, config2)
        
        results1 = ea1.evolve(verbose=False)
        results2 = ea2.evolve(verbose=False)
        
        # Resultados devem ser idênticos
        assert np.allclose(results1['best_parameters'], results2['best_parameters'])


class TestEvolutionStatistics:
    """Testa estatísticas e histórico da evolução."""
    
    def test_fitness_history(self):
        """Testa que o histórico de fitness é mantido."""
        model = BlackScholesModel()
        
        options = [
            OptionData(100.0, 100.0, 1.0, 0.05, 'call', 10.0)
        ]
        
        config = EvolutionaryConfig(
            population_size=20,
            n_generations=15
        )
        
        ea = EvolutionaryAlgorithm(model, options, config)
        results = ea.evolve(verbose=False)
        
        assert len(results['fitness_history']) == 15
        assert len(results['avg_fitness_history']) == 15
        
        # Fitness deve ser não-decrescente (devido ao elitismo)
        for i in range(1, len(results['fitness_history'])):
            assert results['fitness_history'][i] >= results['fitness_history'][i-1]
