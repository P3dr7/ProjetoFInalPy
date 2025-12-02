"""
Módulo de Algoritmo Evolutivo para calibração de parâmetros.

Implementa um Algoritmo Genético completo com seleção, crossover, mutação
e cálculo de fitness baseado em MSE.
"""

from typing import List, Callable, Optional, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from .pricing_models import PricingModel, OptionData, calculate_mse


@dataclass
class EvolutionaryConfig:
    """Configuração do algoritmo evolutivo."""
    
    population_size: int = 100
    n_generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_std: float = 0.1
    tournament_size: int = 3
    elitism_count: int = 2
    random_seed: Optional[int] = 42
    
    def __post_init__(self):
        """Valida a configuração."""
        if self.population_size < 2:
            raise ValueError("Population size deve ser >= 2")
        if not 0 <= self.crossover_rate <= 1:
            raise ValueError("Crossover rate deve estar em [0, 1]")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("Mutation rate deve estar em [0, 1]")
        if self.tournament_size > self.population_size:
            raise ValueError("Tournament size não pode ser maior que population size")


@dataclass
class Individual:
    """Representa um indivíduo (solução candidata) na população."""
    
    genes: np.ndarray  # Vetor de parâmetros
    fitness: float = 0.0
    
    def __lt__(self, other: 'Individual') -> bool:
        """Comparação para ordenação (maior fitness é melhor)."""
        return self.fitness < other.fitness


class EvolutionaryAlgorithm:
    """
    Algoritmo Evolutivo para calibração de parâmetros de modelos de precificação.
    
    Usa fitness = 1 / (1 + MSE) para maximização.
    """
    
    def __init__(
        self,
        model: PricingModel,
        options: List[OptionData],
        config: Optional[EvolutionaryConfig] = None
    ):
        """
        Inicializa o algoritmo evolutivo.
        
        Args:
            model: Modelo de precificação.
            options: Lista de opções para calibração.
            config: Configuração do algoritmo.
        """
        self.model = model
        self.options = options
        self.config = config if config else EvolutionaryConfig()
        
        # Obtém limites dos parâmetros do modelo
        self.bounds = np.array(model.parameter_bounds())
        self.n_parameters = len(self.bounds)
        
        # Estatísticas da evolução
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.best_individual: Optional[Individual] = None
        
        # Seed para reprodutibilidade
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def _initialize_population(self) -> List[Individual]:
        """
        Cria a população inicial com indivíduos aleatórios.
        
        Returns:
            Lista de indivíduos.
        """
        population = []
        
        for _ in range(self.config.population_size):
            # Gera genes aleatórios dentro dos limites
            genes = np.array([
                np.random.uniform(low, high)
                for low, high in self.bounds
            ])
            
            individual = Individual(genes=genes)
            population.append(individual)
        
        return population
    
    def _calculate_fitness(self, individual: Individual) -> float:
        """
        Calcula o fitness de um indivíduo.
        
        Fitness = 1 / (1 + MSE)
        Maior fitness = melhor solução
        
        Args:
            individual: Indivíduo a avaliar.
            
        Returns:
            Valor de fitness.
        """
        try:
            mse = calculate_mse(self.options, self.model, individual.genes)
            # Transforma MSE em fitness (quanto menor MSE, maior fitness)
            fitness = 1.0 / (1.0 + mse)
            return fitness
        except Exception as e:
            # Se houver erro no cálculo, retorna fitness muito baixo
            print(f"Erro ao calcular fitness: {e}")
            return 0.0
    
    def _evaluate_population(self, population: List[Individual]) -> None:
        """
        Avalia o fitness de toda a população.
        
        Args:
            population: População a avaliar.
        """
        for individual in population:
            individual.fitness = self._calculate_fitness(individual)
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Seleção por torneio.
        
        Args:
            population: População para seleção.
            
        Returns:
            Indivíduo selecionado.
        """
        # Seleciona indivíduos aleatórios para o torneio
        tournament = np.random.choice(
            population, 
            size=self.config.tournament_size, 
            replace=False
        )
        
        # Retorna o melhor do torneio
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _roulette_selection(self, population: List[Individual]) -> Individual:
        """
        Seleção por roleta (proporcional ao fitness).
        
        Args:
            population: População para seleção.
            
        Returns:
            Indivíduo selecionado.
        """
        # Calcula fitness total
        total_fitness = sum(ind.fitness for ind in population)
        
        if total_fitness == 0:
            # Se todos têm fitness zero, seleciona aleatoriamente
            return np.random.choice(population)
        
        # Gera número aleatório
        pick = np.random.uniform(0, total_fitness)
        
        # Seleciona baseado na roleta
        current = 0
        for individual in population:
            current += individual.fitness
            if current >= pick:
                return individual
        
        return population[-1]  # Fallback
    
    def _arithmetic_crossover(
        self, 
        parent1: Individual, 
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Crossover aritmético: filho = alpha * p1 + (1-alpha) * p2.
        
        Args:
            parent1: Primeiro pai.
            parent2: Segundo pai.
            
        Returns:
            Tupla com dois filhos.
        """
        alpha = np.random.uniform(0, 1)
        
        child1_genes = alpha * parent1.genes + (1 - alpha) * parent2.genes
        child2_genes = (1 - alpha) * parent1.genes + alpha * parent2.genes
        
        # Garante que os genes estão dentro dos limites
        child1_genes = self._clip_to_bounds(child1_genes)
        child2_genes = self._clip_to_bounds(child2_genes)
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _uniform_crossover(
        self, 
        parent1: Individual, 
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Crossover uniforme: cada gene vem de um dos pais aleatoriamente.
        
        Args:
            parent1: Primeiro pai.
            parent2: Segundo pai.
            
        Returns:
            Tupla com dois filhos.
        """
        mask = np.random.rand(self.n_parameters) < 0.5
        
        child1_genes = np.where(mask, parent1.genes, parent2.genes)
        child2_genes = np.where(mask, parent2.genes, parent1.genes)
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def _gaussian_mutation(self, individual: Individual) -> Individual:
        """
        Mutação gaussiana: adiciona ruído normal aos genes.
        
        Args:
            individual: Indivíduo a mutar.
            
        Returns:
            Indivíduo mutado.
        """
        mutated_genes = individual.genes.copy()
        
        for i in range(self.n_parameters):
            if np.random.rand() < self.config.mutation_rate:
                # Calcula desvio padrão baseado no range do parâmetro
                param_range = self.bounds[i, 1] - self.bounds[i, 0]
                std = self.config.mutation_std * param_range
                
                # Adiciona ruído gaussiano
                mutated_genes[i] += np.random.normal(0, std)
        
        # Garante limites
        mutated_genes = self._clip_to_bounds(mutated_genes)
        
        return Individual(mutated_genes)
    
    def _clip_to_bounds(self, genes: np.ndarray) -> np.ndarray:
        """
        Garante que os genes estão dentro dos limites permitidos.
        
        Args:
            genes: Genes a clipar.
            
        Returns:
            Genes dentro dos limites.
        """
        return np.clip(genes, self.bounds[:, 0], self.bounds[:, 1])
    
    def _create_next_generation(
        self, 
        population: List[Individual]
    ) -> List[Individual]:
        """
        Cria a próxima geração usando seleção, crossover e mutação.
        
        Args:
            population: População atual.
            
        Returns:
            Nova população.
        """
        next_generation = []
        
        # Elitismo: mantém os melhores indivíduos
        if self.config.elitism_count > 0:
            sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
            next_generation.extend(sorted_pop[:self.config.elitism_count])
        
        # Gera o resto da população
        while len(next_generation) < self.config.population_size:
            # Seleção
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if np.random.rand() < self.config.crossover_rate:
                child1, child2 = self._arithmetic_crossover(parent1, parent2)
            else:
                child1, child2 = Individual(parent1.genes.copy()), Individual(parent2.genes.copy())
            
            # Mutação
            child1 = self._gaussian_mutation(child1)
            child2 = self._gaussian_mutation(child2)
            
            next_generation.extend([child1, child2])
        
        # Garante tamanho exato da população
        return next_generation[:self.config.population_size]
    
    def evolve(self, verbose: bool = True) -> Dict[str, any]:
        """
        Executa o algoritmo evolutivo.
        
        Args:
            verbose: Se True, imprime progresso.
            
        Returns:
            Dicionário com resultados da evolução.
        """
        # Inicializa população
        population = self._initialize_population()
        self._evaluate_population(population)
        
        # Evolução
        for generation in range(self.config.n_generations):
            # Cria próxima geração
            population = self._create_next_generation(population)
            self._evaluate_population(population)
            
            # Estatísticas
            best = max(population, key=lambda x: x.fitness)
            avg_fitness = np.mean([ind.fitness for ind in population])
            
            self.best_fitness_history.append(best.fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # Atualiza melhor solução global
            if self.best_individual is None or best.fitness > self.best_individual.fitness:
                self.best_individual = Individual(best.genes.copy(), best.fitness)
            
            # Log
            if verbose and (generation % 10 == 0 or generation == self.config.n_generations - 1):
                mse = 1.0 / best.fitness - 1.0
                print(f"Geração {generation:3d}: "
                      f"Best Fitness = {best.fitness:.6f}, "
                      f"MSE = {mse:.6f}, "
                      f"Avg Fitness = {avg_fitness:.6f}")
        
        # Retorna resultados
        best_mse = 1.0 / self.best_individual.fitness - 1.0
        
        return {
            'best_parameters': self.best_individual.genes,
            'best_fitness': self.best_individual.fitness,
            'best_mse': best_mse,
            'best_rmse': np.sqrt(best_mse),
            'fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'parameter_names': self.model.parameter_names()
        }
    
    def get_parameter_dict(self) -> Dict[str, float]:
        """
        Retorna os melhores parâmetros como dicionário.
        
        Returns:
            Dicionário {nome_parametro: valor}.
        """
        if self.best_individual is None:
            raise ValueError("Evolução ainda não foi executada.")
        
        names = self.model.parameter_names()
        values = self.best_individual.genes
        
        return dict(zip(names, values))
