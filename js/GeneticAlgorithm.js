/**
 * GeneticAlgorithm.js - Genetic Algorithm implementation for training Neural Cellular Automata
 * Uses evolutionary approach instead of backpropagation
 */
class GeneticAlgorithm {
    /**
     * Create a new Genetic Algorithm trainer
     * @param {Grid} grid - The grid instance
     * @param {NeuralNetwork} baseNetwork - Base neural network (for architecture)
     * @param {CellularAutomata} cellularAutomata - CA instance for evaluation
     * @param {Object} config - Configuration object
     * @param {number} config.populationSize - Number of individuals in population (default: 30)
     * @param {number} config.mutationRate - Probability of mutating each weight (default: 0.15)
     * @param {number} config.mutationStrength - Standard deviation for Gaussian mutation noise (default: 0.02)
     * @param {number} config.eliteCount - Number of top performers to preserve unchanged (default: 2)
     */
    constructor(grid, baseNetwork, cellularAutomata, config = {}) {
        if (!grid || !baseNetwork || !cellularAutomata) {
            throw new Error('GeneticAlgorithm requires grid, baseNetwork, and cellularAutomata');
        }
        
        this.grid = grid;
        this.baseNetwork = baseNetwork;
        this.cellularAutomata = cellularAutomata;
        
        // GA parameters
        this.populationSize = config.populationSize || 30;
        this.mutationRate = config.mutationRate || 0.15;
        this.mutationStrength = config.mutationStrength || 0.02;
        this.eliteCount = config.eliteCount || 2;
        
        this.population = []; // Array of {network, fitness, loss}
        this.generation = 0;
        this.isTraining = false;
    }
    
    /**
     * Create a new neural network with the same architecture as base network
     * @returns {NeuralNetwork} New network instance
     */
    _createNetwork() {
        const network = new NeuralNetwork({
            hiddenSize1: this.baseNetwork.hiddenSize1,
            hiddenSize2: this.baseNetwork.hiddenSize2
        });
        network.initialize();
        return network;
    }
    
    /**
     * Initialize population with random networks
     */
    _initializePopulation() {
        this.population = [];
        const baseModel = this.baseNetwork.getModel();
        const baseWeights = baseModel.getWeights();
        
        for (let i = 0; i < this.populationSize; i++) {
            const network = this._createNetwork();
            const model = network.getModel();
            
            if (i === 0) {
                // First individual: use base network weights (cloned)
                const clonedWeights = baseWeights.map(w => w.clone());
                model.setWeights(clonedWeights);
                // Clean up clones after setting (model will keep its own references)
                clonedWeights.forEach(w => w.dispose());
            } else {
                // Other individuals: add random variation to base weights
                const newWeights = baseWeights.map(w => {
                    const noise = tf.randomNormal(w.shape, 0, 0.1);
                    const noisy = w.add(noise);
                    return noisy;
                });
                model.setWeights(newWeights);
                // Clean up
                newWeights.forEach(w => w.dispose());
            }
            
            this.population.push({
                network: network,
                fitness: null,
                loss: null
            });
        }
    }
    
    /**
     * Reset grid to seed cell at center
     */
    _resetToSeedCell() {
        this.grid.clear();
        const centerX = Math.floor(this.grid.width / 2);
        const centerY = Math.floor(this.grid.height / 2);
        this.grid.setCell(centerX, centerY, true);
    }
    
    /**
     * Evaluate fitness of a network
     * @param {NeuralNetwork} network - Network to evaluate
     * @param {Array<Array<boolean>>} targetShape - 5×5 target shape
     * @param {number} genSteps - Number of CA steps to run
     * @returns {Object} {fitness, loss}
     */
    _evaluateFitness(network, targetShape, genSteps) {
        // Reset grid to seed cell
        this._resetToSeedCell();
        
        // Temporarily replace CA's network with this one
        const originalNetwork = this.cellularAutomata.neuralNetwork;
        this.cellularAutomata.neuralNetwork = network;
        
        // Run CA for genSteps
        for (let step = 0; step < genSteps; step++) {
            this.cellularAutomata.update();
        }
        
        // Compute loss (lower is better)
        const loss = this._computeLoss(targetShape);
        
        // Convert loss to fitness (higher is better)
        // Use inverse with small epsilon to avoid division by zero
        const fitness = 1.0 / (loss + 0.0001);
        
        // Restore original network
        this.cellularAutomata.neuralNetwork = originalNetwork;
        
        return { fitness, loss };
    }
    
    /**
     * Compute loss between current grid state and target shape
     * @param {Array<Array<boolean>>} targetShape - 5×5 boolean array
     * @returns {number} Loss value (mean squared error)
     */
    _computeLoss(targetShape) {
        if (!targetShape || targetShape.length !== 5 || targetShape[0].length !== 5) {
            throw new Error('Target shape must be a 5×5 boolean array');
        }
        
        // Extract center 5×5 region from 9×9 grid
        const centerX = Math.floor(this.grid.width / 2) - 2;
        const centerY = Math.floor(this.grid.height / 2) - 2;
        
        let totalLoss = 0;
        let cellCount = 0;
        
        for (let ty = 0; ty < 5; ty++) {
            for (let tx = 0; tx < 5; tx++) {
                const gridX = centerX + tx;
                const gridY = centerY + ty;
                
                const cell = this.grid.getCell(gridX, gridY);
                const targetValue = targetShape[ty][tx] ? 1.0 : 0.0;
                const actualValue = cell.on ? 1.0 : 0.0;
                
                // Mean squared error
                const error = targetValue - actualValue;
                totalLoss += error * error;
                cellCount++;
            }
        }
        
        return totalLoss / cellCount;
    }
    
    /**
     * Crossover: create child from two parents using uniform crossover
     * @param {NeuralNetwork} parent1 - First parent network
     * @param {NeuralNetwork} parent2 - Second parent network
     * @returns {NeuralNetwork} Child network
     */
    _crossover(parent1, parent2) {
        const child = this._createNetwork();
        const childModel = child.getModel();
        const p1Weights = parent1.getModel().getWeights();
        const p2Weights = parent2.getModel().getWeights();
        
        const childWeights = p1Weights.map((w1, i) => {
            const w2 = p2Weights[i];
            
            // Uniform crossover: randomly pick from each parent
            const mask = tf.randomUniform(w1.shape, 0, 1);
            const threshold = tf.scalar(0.5);
            const fromP1 = mask.greater(threshold);
            const fromP2 = mask.lessEqual(threshold);
            
            const childW = w1.mul(fromP1.cast('float32'))
                            .add(w2.mul(fromP2.cast('float32')));
            
            // Clean up intermediate tensors
            mask.dispose();
            threshold.dispose();
            fromP1.dispose();
            fromP2.dispose();
            
            return childW;
        });
        
        childModel.setWeights(childWeights);
        
        // Note: setWeights takes ownership of childWeights tensors
        // The model will manage them, so we don't dispose them here
        // The original p1Weights and p2Weights belong to parent networks and should never be disposed
        
        return child;
    }
    
    /**
     * Mutate a network's weights
     * @param {NeuralNetwork} network - Network to mutate
     */
    _mutate(network) {
        const model = network.getModel();
        const weights = model.getWeights();
        
        const mutatedWeights = weights.map(w => {
            // Create mutation mask (mutate mutationRate% of weights)
            const mutationMask = tf.randomUniform(w.shape, 0, 1);
            const shouldMutate = mutationMask.less(tf.scalar(this.mutationRate));
            
            // Generate Gaussian noise
            const noise = tf.randomNormal(w.shape, 0, this.mutationStrength);
            
            // Apply mutation only where mask indicates
            const mutated = w.add(noise.mul(shouldMutate.cast('float32')));
            
            // Clean up
            mutationMask.dispose();
            shouldMutate.dispose();
            noise.dispose();
            
            return mutated;
        });
        
        model.setWeights(mutatedWeights);
        
        // Note: setWeights takes ownership of mutatedWeights tensors
        // The model will manage them, so we don't dispose them here
        // The original weights belong to the network and should never be disposed
    }
    
    /**
     * Train using genetic algorithm
     * @param {Array<Array<boolean>>} targetShape - 5×5 target shape
     * @param {number} numGenerations - Number of generations to evolve
     * @param {number} genSteps - Number of CA steps for fitness evaluation
     * @param {Function} progressCallback - Optional callback (generation, bestLoss, shouldContinue)
     * @returns {Promise<Array<number>>} Array of best loss values per generation
     */
    async train(targetShape, numGenerations = 100, genSteps = 50, progressCallback = null) {
        if (!targetShape || targetShape.length !== 5 || targetShape[0].length !== 5) {
            throw new Error('Target shape must be a 5×5 boolean array');
        }
        
        this.isTraining = true;
        this.generation = 0;
        const lossHistory = [];
        
        // Initialize population
        console.log(`Initializing population of ${this.populationSize}...`);
        this._initializePopulation();
        
        for (let gen = 0; gen < numGenerations; gen++) {
            if (!this.isTraining) break;
            
            this.generation = gen + 1;
            console.log(`Generation ${this.generation}/${numGenerations}`);
            
            // Evaluate fitness for all networks
            console.log('Evaluating fitness...');
            for (let i = 0; i < this.population.length; i++) {
                const individual = this.population[i];
                const result = this._evaluateFitness(individual.network, targetShape, genSteps);
                individual.fitness = result.fitness;
                individual.loss = result.loss;
            }
            
            // Sort by fitness (descending - higher fitness is better)
            this.population.sort((a, b) => b.fitness - a.fitness);
            
            // Get best loss (from top performer)
            const bestLoss = this.population[0].loss;
            lossHistory.push(bestLoss);
            
            console.log(`Generation ${this.generation}: Best loss = ${bestLoss.toFixed(6)}`);
            
            // Call progress callback
            if (progressCallback) {
                const shouldContinue = progressCallback(this.generation, bestLoss, true) !== false;
                if (!shouldContinue) {
                    this.isTraining = false;
                    break;
                }
            }
            
            // Create next generation (except on last iteration)
            if (gen < numGenerations - 1) {
                const nextGeneration = [];
                
                // Elite: keep top performers unchanged
                for (let i = 0; i < this.eliteCount; i++) {
                    const elite = this.population[i];
                    const eliteNetwork = this._createNetwork();
                    const eliteWeights = elite.network.getModel().getWeights();
                    // Clone weights before setting (setWeights takes ownership)
                    const clonedEliteWeights = eliteWeights.map(w => w.clone());
                    eliteNetwork.getModel().setWeights(clonedEliteWeights);
                    // Note: setWeights takes ownership of the cloned weights, so we don't dispose them
                    // The original eliteWeights belong to elite.network and should not be disposed
                    nextGeneration.push({
                        network: eliteNetwork,
                        fitness: null,
                        loss: null
                    });
                }
                
                // Breed rest of population from top 2
                const parent1 = this.population[0].network;
                const parent2 = this.population[1].network;
                
                for (let i = this.eliteCount; i < this.populationSize; i++) {
                    // Create child from two parents
                    const child = this._crossover(parent1, parent2);
                    
                    // Mutate child
                    this._mutate(child);
                    
                    nextGeneration.push({
                        network: child,
                        fitness: null,
                        loss: null
                    });
                }
                
                // Dispose old population (except elites which we cloned)
                for (let i = this.eliteCount; i < this.population.length; i++) {
                    this.population[i].network.dispose();
                }
                
                this.population = nextGeneration;
            }
            
            // Small delay for UI updates
            await new Promise(resolve => setTimeout(resolve, 10));
        }
        
        // Don't dispose population networks here - they may still be referenced
        // The best network will be used, and others will be cleaned up when trainer is disposed
        this.isTraining = false;
        return lossHistory;
    }
    
    /**
     * Get the best network from the population
     * @returns {NeuralNetwork} Best performing network
     */
    getBestNetwork() {
        if (this.population.length === 0) {
            return null;
        }
        
        // Sort by fitness to ensure best is first
        this.population.sort((a, b) => b.fitness - a.fitness);
        return this.population[0].network;
    }
    
    /**
     * Stop training
     */
    stopTraining() {
        this.isTraining = false;
    }
    
    /**
     * Get current training status
     * @returns {boolean} True if currently training
     */
    getIsTraining() {
        return this.isTraining;
    }
    
    /**
     * Dispose of all resources
     */
    dispose() {
        this.stopTraining();
        if (this.population) {
            this.population.forEach(ind => {
                if (ind.network) ind.network.dispose();
            });
        }
        this.population = [];
    }
}

