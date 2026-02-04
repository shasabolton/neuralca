/**
 * Trainer.js - Training system for Neural Cellular Automata
 * Uses Genetic Algorithm for training instead of backpropagation
 */
class Trainer {
    /**
     * Create a new Trainer instance
     * @param {Grid} grid - The grid instance to train on
     * @param {NeuralNetwork} neuralNetwork - The neural network to train
     * @param {CellularAutomata} cellularAutomata - The CA instance (required for training)
     * @param {Object} config - Configuration object
     * @param {number} config.populationSize - Population size for GA (default: 30)
     * @param {number} config.mutationRate - Mutation rate for GA (default: 0.15)
     * @param {number} config.mutationStrength - Mutation strength for GA (default: 0.02)
     * @param {number} config.eliteCount - Number of elite individuals to preserve (default: 2)
     */
    constructor(grid, neuralNetwork, cellularAutomata = null, config = {}) {
        if (!grid || !neuralNetwork) {
            throw new Error('Trainer requires both grid and neuralNetwork');
        }
        
        this.grid = grid;
        this.neuralNetwork = neuralNetwork;
        this.cellularAutomata = cellularAutomata;
        
        // GA parameters
        this.populationSize = config.populationSize || 30;
        this.mutationRate = config.mutationRate || 0.15;
        this.mutationStrength = config.mutationStrength || 0.02;
        this.eliteCount = config.eliteCount || 2;
        
        this.isTraining = false;
        this.trainingStep = 0;
        this.lossHistory = [];
        
        // Initialize genetic algorithm
        this.geneticAlgorithm = new GeneticAlgorithm(
            grid,
            neuralNetwork,
            cellularAutomata,
            {
                populationSize: this.populationSize,
                mutationRate: this.mutationRate,
                mutationStrength: this.mutationStrength,
                eliteCount: this.eliteCount
            }
        );
    }
    
    
    /**
     * Reset grid to a single seed cell at the center
     */
    _resetToSeedCell() {
        this.grid.clear();
        const centerX = Math.floor(this.grid.width / 2);
        const centerY = Math.floor(this.grid.height / 2);
        this.grid.setCell(centerX, centerY, true);
    }
    
    /**
     * Compute loss between current grid state and target shape
     * The target shape is 5×5, so we extract the center 5×5 region from the 9×9 grid
     * @param {Array<Array<boolean>>} targetShape - 5×5 boolean array representing target shape
     * @returns {number} Loss value (mean squared error)
     */
    computeLoss(targetShape) {
        if (!targetShape || targetShape.length !== 5 || targetShape[0].length !== 5) {
            throw new Error('Target shape must be a 5×5 boolean array');
        }
        
        // Extract center 5×5 region from 9×9 grid (centered at pixel 4,4)
        // 5×5 region spans from (2,2) to (6,6) - symmetric about center
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
     * Compute loss tensor for a grid tensor (differentiable)
     * Compares center 5×5 region to target shape
     * NOTE: Do NOT dispose intermediate tensors - they're needed for gradients
     * @param {tf.Tensor} gridTensor - Grid state tensor [height, width, 3]
     * @param {tf.Tensor} targetTensor - Target tensor [5, 5]
     * @returns {tf.Scalar} Loss tensor
     */
    _computeLossTensor(gridTensor, targetTensor) {
        // Extract center 5×5 region from 9×9 grid (centered at pixel 4,4)
        const centerX = Math.floor(this.grid.width / 2) - 2;
        const centerY = Math.floor(this.grid.height / 2) - 2;
        
        // Extract center 5×5 region from grid tensor
        // Get only the on/off channel (channel 0)
        const centerRegion = gridTensor.slice([centerY, centerX, 0], [5, 5, 1]);
        const centerOnOff = centerRegion.reshape([5, 5]);
        
        // Apply sigmoid to get probabilities (in case values aren't already in [0,1])
        const probabilities = tf.sigmoid(centerOnOff);
        
        // Use binary cross-entropy instead of MSE for better gradient flow
        // Clip probabilities to avoid log(0)
        const clippedProbs = probabilities.clipByValue(1e-7, 1 - 1e-7);
        
        // Binary cross-entropy: -[y*log(p) + (1-y)*log(1-p)]
        const bce = targetTensor.mul(clippedProbs.log())
            .add(targetTensor.mul(-1).add(1).mul(clippedProbs.mul(-1).add(1).log()))
            .mul(-1)
            .mean();
        
        // Add penalty if all cells are off when target has some on
        // This prevents the model from learning the "all off" solution
        const targetSum = targetTensor.sum();
        const probSum = probabilities.sum();
        const minProbSum = targetSum.mul(0.5); // At least 50% of target sum
        const sumPenalty = tf.maximum(0, minProbSum.sub(probSum)).mul(0.1);
        
        // Don't dispose intermediate tensors - they're needed for gradient computation
        // TensorFlow will manage cleanup after gradients are computed
        
        return bce.add(sumPenalty);
    }
    
    /**
     * Train the neural network using Genetic Algorithm
     * 
     * @param {Array<Array<boolean>>} targetShape - 5×5 boolean array
     * @param {number} numGenerations - Number of generations to evolve
     * @param {Function} progressCallback - Optional callback (generation, loss, shouldContinue)
     * @returns {Promise<Array<number>>} Array of loss values
     */
    async train(targetShape, numGenerations = 100, progressCallback = null) {
        if (!this.neuralNetwork.isInitialized) {
            throw new Error('Neural network not initialized. Call neuralNetwork.initialize() first.');
        }
        
        if (!targetShape || targetShape.length !== 5 || targetShape[0].length !== 5) {
            throw new Error('Target shape must be a 5×5 boolean array');
        }
        
        if (!this.cellularAutomata) {
            throw new Error('CellularAutomata instance required for training');
        }
        
        // Get genSteps from UI
        const genStepsDropdown = document.getElementById('genSteps');
        const genSteps = genStepsDropdown ? parseInt(genStepsDropdown.value, 10) : 50;
        
        // Update GA parameters from UI if available
        const populationInput = document.getElementById('populationSize');
        const mutationRateInput = document.getElementById('mutationRate');
        const mutationStrengthInput = document.getElementById('mutationStrength');
        const eliteCountInput = document.getElementById('eliteCount');
        
        if (populationInput) {
            this.populationSize = parseInt(populationInput.value, 10);
        }
        if (mutationRateInput) {
            this.mutationRate = parseFloat(mutationRateInput.value);
        }
        if (mutationStrengthInput) {
            this.mutationStrength = parseFloat(mutationStrengthInput.value);
        }
        if (eliteCountInput) {
            this.eliteCount = parseInt(eliteCountInput.value, 10);
        }
        
        // Recreate genetic algorithm with updated parameters
        if (this.geneticAlgorithm) {
            this.geneticAlgorithm.dispose();
        }
        this.geneticAlgorithm = new GeneticAlgorithm(
            this.grid,
            this.neuralNetwork,
            this.cellularAutomata,
            {
                populationSize: this.populationSize,
                mutationRate: this.mutationRate,
                mutationStrength: this.mutationStrength,
                eliteCount: this.eliteCount
            }
        );
        
        this.isTraining = true;
        
        // Train using genetic algorithm
        const losses = await this.geneticAlgorithm.train(
            targetShape,
            numGenerations,
            genSteps,
            (generation, bestLoss, shouldContinue) => {
                this.trainingStep = generation;
                this.lossHistory.push(bestLoss);
                
                // Update main network with best performer after each generation
                // This allows user to stop and use the current best
                const bestNetwork = this.geneticAlgorithm.getBestNetwork();
                if (bestNetwork) {
                    const bestWeights = bestNetwork.getModel().getWeights();
                    // Clone weights before setting (setWeights takes ownership)
                    const clonedWeights = bestWeights.map(w => w.clone());
                    this.neuralNetwork.getModel().setWeights(clonedWeights);
                    // Dispose cloned weights after setting (setWeights takes ownership, but we cloned so we need to dispose our clones)
                    // Actually, setWeights takes ownership, so we don't need to dispose. But to be safe, let's not dispose the originals.
                    // The originals belong to bestNetwork and should not be disposed.
                }
                
                if (progressCallback) {
                    return progressCallback(generation, bestLoss, shouldContinue);
                }
                return shouldContinue;
            }
        );
        
        // Ensure main network is updated with best performer (in case training completed)
        const bestNetwork = this.geneticAlgorithm.getBestNetwork();
        if (bestNetwork) {
            const bestWeights = bestNetwork.getModel().getWeights();
            // Clone weights before setting (setWeights takes ownership)
            const clonedWeights = bestWeights.map(w => w.clone());
            this.neuralNetwork.getModel().setWeights(clonedWeights);
            // Note: setWeights takes ownership of the cloned weights
            // The original bestWeights belong to bestNetwork and should not be disposed
        }
        
        this.isTraining = false;
        return losses;
    }
    
    
    /**
     * Stop training (if currently training)
     */
    stopTraining() {
        this.isTraining = false;
        if (this.geneticAlgorithm) {
            this.geneticAlgorithm.stopTraining();
        }
    }
    
    /**
     * Get current training status
     * @returns {boolean} True if currently training
     */
    getIsTraining() {
        return this.isTraining;
    }
    
    /**
     * Get loss history
     * @returns {Array<number>} Array of loss values from training
     */
    getLossHistory() {
        return [...this.lossHistory];
    }
    
    /**
     * Clear loss history
     */
    clearLossHistory() {
        this.lossHistory = [];
        this.trainingStep = 0;
    }
    
    
    /**
     * Dispose of resources
     */
    dispose() {
        this.stopTraining();
        if (this.geneticAlgorithm) {
            this.geneticAlgorithm.dispose();
            this.geneticAlgorithm = null;
        }
        this.lossHistory = [];
    }
}
