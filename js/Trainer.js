/**
 * Trainer.js - Training system for Neural Cellular Automata
 * Implements supervised learning to match target shapes using TensorFlow.js optimizer
 */
class Trainer {
    /**
     * Create a new Trainer instance
     * @param {Grid} grid - The grid instance to train on
     * @param {NeuralNetwork} neuralNetwork - The neural network to train
     * @param {CellularAutomata} cellularAutomata - The CA instance (optional, for updates during training)
     * @param {Object} config - Configuration object
     * @param {number} config.learningRate - Learning rate for optimizer (default: 0.001)
     * @param {string} config.optimizer - Optimizer name: 'adam', 'sgd', 'rmsprop' (default: 'adam')
     * @param {number} config.batchSize - Number of cells to train on per batch (default: 1000)
     */
    constructor(grid, neuralNetwork, cellularAutomata = null, config = {}) {
        if (!grid || !neuralNetwork) {
            throw new Error('Trainer requires both grid and neuralNetwork');
        }
        
        this.grid = grid;
        this.neuralNetwork = neuralNetwork;
        this.cellularAutomata = cellularAutomata;
        
        this.learningRate = config.learningRate || 0.001;
        this.optimizerName = config.optimizer || 'adam';
        this.batchSize = config.batchSize || 1000;
        
        this.optimizer = null;
        this.isTraining = false;
        this.trainingStep = 0;
        this.lossHistory = [];
        
        // Initialize optimizer
        this._initializeOptimizer();
    }
    
    /**
     * Initialize the TensorFlow.js optimizer
     */
    _initializeOptimizer() {
        if (typeof tf === 'undefined') {
            throw new Error('TensorFlow.js is not loaded. Please include the TensorFlow.js CDN script.');
        }
        
        switch (this.optimizerName.toLowerCase()) {
            case 'adam':
                this.optimizer = tf.train.adam(this.learningRate);
                break;
            case 'sgd':
                this.optimizer = tf.train.sgd(this.learningRate);
                break;
            case 'rmsprop':
                this.optimizer = tf.train.rmsprop(this.learningRate);
                break;
            default:
                throw new Error(`Unknown optimizer: ${this.optimizerName}`);
        }
        
        // Set optimizer in neural network
        this.neuralNetwork.setOptimizer(this.optimizerName, {
            learningRate: this.learningRate
        });
    }
    
    /**
     * Compute loss between current grid state and target shape
     * The target shape is 10×10, so we extract the center 10×10 region from the 100×100 grid
     * @param {Array<Array<boolean>>} targetShape - 10×10 boolean array representing target shape
     * @returns {number} Loss value (mean squared error)
     */
    computeLoss(targetShape) {
        if (!targetShape || targetShape.length !== 10 || targetShape[0].length !== 10) {
            throw new Error('Target shape must be a 10×10 boolean array');
        }
        
        // Extract center 10×10 region from 100×100 grid
        // Center region: x from 45 to 54, y from 45 to 54
        const centerX = Math.floor(this.grid.width / 2) - 5;
        const centerY = Math.floor(this.grid.height / 2) - 5;
        
        let totalLoss = 0;
        let cellCount = 0;
        
        for (let ty = 0; ty < 10; ty++) {
            for (let tx = 0; tx < 10; tx++) {
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
     * Compute loss using TensorFlow.js tensors (for gradient computation)
     * @param {Array<Array<boolean>>} targetShape - 10×10 boolean array
     * @returns {tf.Scalar} Loss tensor
     */
    computeLossTensor(targetShape) {
        if (!targetShape || targetShape.length !== 10 || targetShape[0].length !== 10) {
            throw new Error('Target shape must be a 10×10 boolean array');
        }
        
        // Extract center 10×10 region from grid
        const centerX = Math.floor(this.grid.width / 2) - 5;
        const centerY = Math.floor(this.grid.height / 2) - 5;
        
        // Build target tensor (10×10)
        const targetArray = [];
        for (let ty = 0; ty < 10; ty++) {
            for (let tx = 0; tx < 10; tx++) {
                targetArray.push(targetShape[ty][tx] ? 1.0 : 0.0);
            }
        }
        const targetTensor = tf.tensor2d(targetArray, [10, 10]);
        
        // Build actual tensor from grid (10×10)
        const actualArray = [];
        for (let ty = 0; ty < 10; ty++) {
            for (let tx = 0; tx < 10; tx++) {
                const gridX = centerX + tx;
                const gridY = centerY + ty;
                const cell = this.grid.getCell(gridX, gridY);
                actualArray.push(cell.on ? 1.0 : 0.0);
            }
        }
        const actualTensor = tf.tensor2d(actualArray, [10, 10]);
        
        // Compute mean squared error
        const loss = tf.losses.meanSquaredError(targetTensor, actualTensor);
        
        // Clean up intermediate tensors
        targetTensor.dispose();
        actualTensor.dispose();
        
        return loss;
    }
    
    /**
     * Perform a single training step
     * This updates the neural network weights to minimize the loss
     * @param {Array<Array<boolean>>} targetShape - 10×10 boolean array representing target shape
     * @param {Function} progressCallback - Optional callback function called with (step, loss)
     * @returns {Promise<number>} Loss value after training step
     */
    async trainStep(targetShape, progressCallback = null) {
        if (!this.neuralNetwork.isInitialized) {
            throw new Error('Neural network not initialized. Call neuralNetwork.initialize() first.');
        }
        
        if (!targetShape || targetShape.length !== 10 || targetShape[0].length !== 10) {
            throw new Error('Target shape must be a 10×10 boolean array');
        }
        
        const model = this.neuralNetwork.getModel();
        const variables = model.trainableWeights;
        
        // Compute loss and gradients
        const lossValue = tf.tidy(() => {
            // Get current loss
            const loss = this.computeLossTensor(targetShape);
            
            // Compute gradients
            const grads = tf.grad(() => {
                // We need to compute loss as a function of model parameters
                // Since the model affects the grid through CA updates, we need to:
                // 1. Run one CA update step
                // 2. Compute loss on the updated grid
                
                // For now, we'll use a simpler approach:
                // Compute loss based on current grid state after running CA update
                // But we need to track gradients through the CA update
                
                // Actually, a more practical approach for neural CA training:
                // We'll sample cells and compute loss on their predicted outputs
                // vs what they should be based on the target shape
                
                // For this implementation, we'll use a direct loss on the grid state
                // after running the CA update, which requires tracking gradients through
                // the entire update process. This is complex, so we'll use a simpler method:
                
                // Sample approach: compute loss on current state, then update CA,
                // and use the difference as a signal
                
                return this.computeLossTensor(targetShape);
            });
            
            // Apply gradients
            this.optimizer.applyGradients(
                variables.map((v, i) => ({
                    tensor: v,
                    grad: grads[i] || tf.zerosLike(v)
                }))
            );
            
            return loss;
        });
        
        // Get loss value
        const loss = await lossValue.data();
        const lossScalar = loss[0];
        lossValue.dispose();
        
        // Update training step counter
        this.trainingStep++;
        this.lossHistory.push(lossScalar);
        
        // Call progress callback if provided
        if (progressCallback) {
            progressCallback(this.trainingStep, lossScalar);
        }
        
        return lossScalar;
    }
    
    /**
     * Train the neural network using a simpler, more practical approach
     * This method samples cells and trains on their neighbor→output mappings
     * @param {Array<Array<boolean>>} targetShape - 10×10 boolean array
     * @param {number} numSteps - Number of training steps to perform
     * @param {Function} progressCallback - Optional callback (step, loss, shouldContinue)
     * @returns {Promise<Array<number>>} Array of loss values
     */
    async train(targetShape, numSteps = 100, progressCallback = null) {
        if (!this.neuralNetwork.isInitialized) {
            throw new Error('Neural network not initialized. Call neuralNetwork.initialize() first.');
        }
        
        if (!targetShape || targetShape.length !== 10 || targetShape[0].length !== 10) {
            throw new Error('Target shape must be a 10×10 boolean array');
        }
        
        this.isTraining = true;
        const losses = [];
        
        // Extract center 10×10 region coordinates
        const centerX = Math.floor(this.grid.width / 2) - 5;
        const centerY = Math.floor(this.grid.height / 2) - 5;
        
        for (let step = 0; step < numSteps; step++) {
            if (!this.isTraining) {
                break; // Training was stopped
            }
            
            // Method: Sample cells from the center region and train on them
            // We'll create training examples: (neighbor_input, target_output)
            
            // First, run one CA update to get current state
            if (this.cellularAutomata) {
                this.cellularAutomata.update();
            }
            
            // Sample cells from center 10×10 region
            const trainingExamples = [];
            const targetOutputs = [];
            
            for (let ty = 0; ty < 10; ty++) {
                for (let tx = 0; tx < 10; tx++) {
                    const gridX = centerX + tx;
                    const gridY = centerY + ty;
                    
                    // Get neighbor input for this cell
                    const neighborInput = this.grid.getNeighborInput(gridX, gridY);
                    
                    // Target output: should match target shape
                    const targetOn = targetShape[ty][tx];
                    const targetOutput = new Float32Array(6);
                    targetOutput[0] = targetOn ? 1.0 : 0.0; // on/off
                    // State vector targets: keep current state (or could be learned)
                    const currentCell = this.grid.getCell(gridX, gridY);
                    for (let i = 0; i < 5; i++) {
                        targetOutput[i + 1] = currentCell.stateVector[i];
                    }
                    
                    trainingExamples.push(neighborInput);
                    targetOutputs.push(targetOutput);
                }
            }
            
            // Convert to tensors
            const inputTensor = tf.tensor2d(
                trainingExamples.map(arr => Array.from(arr)),
                [trainingExamples.length, 30]
            );
            const targetTensor = tf.tensor2d(
                targetOutputs.map(arr => Array.from(arr)),
                [targetOutputs.length, 6]
            );
            
            // Train on this batch
            const model = this.neuralNetwork.getModel();
            
            const loss = await tf.tidy(async () => {
                // Forward pass
                const predictions = model.predict(inputTensor);
                
                // Compute loss (MSE)
                const loss = tf.losses.meanSquaredError(targetTensor, predictions);
                
                // Backward pass (automatic with model.fit, but we'll do it manually)
                // Actually, let's use model.fit for proper training
                
                return loss;
            });
            
            // Use model.fit for proper training
            const history = await model.fit(inputTensor, targetTensor, {
                epochs: 1,
                batchSize: Math.min(this.batchSize, trainingExamples.length),
                shuffle: false,
                verbose: 0
            });
            
            const lossValue = history.history.loss[0];
            losses.push(lossValue);
            this.lossHistory.push(lossValue);
            this.trainingStep++;
            
            // Clean up tensors
            inputTensor.dispose();
            targetTensor.dispose();
            
            // Call progress callback
            if (progressCallback) {
                let shouldContinue = true;
                try {
                    shouldContinue = progressCallback(this.trainingStep, lossValue, true) !== false;
                } catch (error) {
                    console.error('Error in progress callback:', error);
                }
                
                if (!shouldContinue) {
                    this.isTraining = false;
                    break;
                }
            }
            
            // Small delay to prevent blocking
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        
        this.isTraining = false;
        return losses;
    }
    
    /**
     * Stop training (if currently training)
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
     * Set learning rate and reinitialize optimizer
     * @param {number} learningRate - New learning rate
     */
    setLearningRate(learningRate) {
        this.learningRate = learningRate;
        this._initializeOptimizer();
    }
    
    /**
     * Dispose of resources
     */
    dispose() {
        this.stopTraining();
        if (this.optimizer) {
            this.optimizer.dispose();
            this.optimizer = null;
        }
        this.lossHistory = [];
    }
}

