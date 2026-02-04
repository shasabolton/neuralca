/**
 * NeuralNetwork.js - TensorFlow.js model wrapper for Neural Cellular Automata
 * Architecture: 15 input → hidden1 (ReLU) → hidden2 (ReLU) → 3 output
 * Input: 15 values (5 neighbors × 3 values: 1 on/off + 2 state vector floats)
 * Output: 3 values (1 on/off + 2 state vector floats)
 */
class NeuralNetwork {
    /**
     * Create a new neural network
     * @param {Object} config - Configuration object
     * @param {number} config.hiddenSize1 - Size of first hidden layer (default: 64)
     * @param {number} config.hiddenSize2 - Size of second hidden layer (default: 128)
     */
    constructor(config = {}) {
        this.hiddenSize1 = config.hiddenSize1 || 16;
        this.hiddenSize2 = config.hiddenSize2 || 16;
        this.model = null;
        this.isInitialized = false;
    }
    
    /**
     * Initialize the neural network model
     * Creates a TensorFlow.js sequential model with the specified architecture
     */
    initialize() {
        if (this.isInitialized) {
            console.warn('Neural network already initialized');
            return;
        }
        
        // Check if TensorFlow.js is available
        if (typeof tf === 'undefined') {
            throw new Error('TensorFlow.js is not loaded. Please include the TensorFlow.js CDN script.');
        }
        
        // Create sequential model
        this.model = tf.sequential({
            layers: [
                // Input layer: 15 neurons (5 neighbors × 3 values)
                tf.layers.dense({
                    inputShape: [15],
                    units: this.hiddenSize1,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.0001}), // L2 regularization
                    name: 'hidden1'
                }),
                
                // Second hidden layer
                tf.layers.dense({
                    units: this.hiddenSize2,
                    activation: 'relu',
                    kernelRegularizer: tf.regularizers.l2({l2: 0.0001}), // L2 regularization
                    name: 'hidden2'
                }),
                
                // Dropout layer to prevent overfitting (only during training)
                tf.layers.dropout({
                    rate: 0.1, // 10% dropout
                    name: 'dropout'
                }),
                
                // Output layer: 3 neurons (1 on/off + 2 state vector floats)
                tf.layers.dense({
                    units: 3,
                    activation: 'linear', // We'll apply sigmoid to first output separately
                    name: 'output'
                })
            ]
        });
        
        // Compile the model (optimizer will be set during training)
        // For now, we'll use a default optimizer
        this.model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });
        
        this.isInitialized = true;
        console.log(`Neural network initialized: 15 → ${this.hiddenSize1} → ${this.hiddenSize2} → 3`);
    }
    
    /**
     * Forward pass: predict new cell state from neighbor states
     * @param {Float32Array|Array|tf.Tensor} input - 15-element input array (5 neighbors × 3 values)
     * @returns {Object} { on: boolean, stateVector: Float32Array(2) }
     */
    predict(input) {
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized. Call initialize() first.');
        }
        
        // Convert input to tensor if needed
        let inputTensor;
        if (input instanceof tf.Tensor) {
            inputTensor = input;
        } else {
            // Ensure input is a 2D tensor [batch_size, 15]
            const inputArray = Array.isArray(input) ? input : Array.from(input);
            inputTensor = tf.tensor2d([inputArray], [1, 15]);
        }
        
        // Run forward pass
        const outputTensor = this.model.predict(inputTensor);
        
        // Extract values synchronously
        const outputValues = outputTensor.dataSync();
        
        // Clean up tensors
        inputTensor.dispose();
        outputTensor.dispose();
        
        // Process output: first value is on/off (apply sigmoid), rest are state vector
        const onValue = 1 / (1 + Math.exp(-outputValues[0])); // Sigmoid
        const on = onValue > 0.5;
        
        // Apply tanh to state vector to keep values in [-1, 1] range
        const stateVector = new Float32Array(2);
        for (let i = 0; i < 2; i++) {
            stateVector[i] = Math.tanh(outputValues[i + 1]);
        }
        
        return {
            on: on,
            stateVector: stateVector
        };
    }
    
    /**
     * Batch predict: predict multiple cell states at once (more efficient)
     * @param {Array<Float32Array>|tf.Tensor} inputs - Array of 15-element input arrays or a 2D tensor
     * @returns {Array<Object>} Array of { on: boolean, stateVector: Float32Array(2) }
     */
    predictBatch(inputs) {
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized. Call initialize() first.');
        }
        
        // Convert inputs to tensor if needed
        let inputTensor;
        if (inputs instanceof tf.Tensor) {
            inputTensor = inputs;
        } else {
            // Convert array of arrays to 2D tensor [batch_size, 15]
            const inputArray = inputs.map(arr => 
                Array.isArray(arr) ? arr : Array.from(arr)
            );
            inputTensor = tf.tensor2d(inputArray, [inputs.length, 15]);
        }
        
        // Run forward pass
        const outputTensor = this.model.predict(inputTensor);
        
        // Extract values synchronously
        const outputValues = outputTensor.dataSync();
        const batchSize = inputTensor.shape[0];
        
        // Clean up tensors
        inputTensor.dispose();
        outputTensor.dispose();
        
        // Process outputs
        const results = [];
        for (let i = 0; i < batchSize; i++) {
            const baseIdx = i * 3;
            
            // Apply sigmoid to on/off value
            const onValue = 1 / (1 + Math.exp(-outputValues[baseIdx]));
            const on = onValue > 0.5;
            
            // Apply tanh to state vector
            const stateVector = new Float32Array(2);
            for (let j = 0; j < 2; j++) {
                stateVector[j] = Math.tanh(outputValues[baseIdx + j + 1]);
            }
            
            results.push({
                on: on,
                stateVector: stateVector
            });
        }
        
        return results;
    }
    
    /**
     * Get the underlying TensorFlow.js model
     * @returns {tf.Sequential} The model
     */
    getModel() {
        return this.model;
    }
    
    /**
     * Set the optimizer for training
     * @param {tf.Optimizer|string} optimizer - TensorFlow optimizer or optimizer name
     * @param {Object} config - Optimizer configuration
     */
    setOptimizer(optimizer, config = {}) {
        if (!this.isInitialized) {
            throw new Error('Neural network not initialized. Call initialize() first.');
        }
        
        let opt;
        if (typeof optimizer === 'string') {
            switch (optimizer.toLowerCase()) {
                case 'adam':
                    opt = tf.train.adam(config.learningRate || 0.001);
                    break;
                case 'sgd':
                    opt = tf.train.sgd(config.learningRate || 0.01);
                    break;
                case 'rmsprop':
                    opt = tf.train.rmsprop(config.learningRate || 0.001);
                    break;
                default:
                    throw new Error(`Unknown optimizer: ${optimizer}`);
            }
        } else {
            opt = optimizer;
        }
        
        this.model.compile({
            optimizer: opt,
            loss: config.loss || 'meanSquaredError'
        });
    }
    
    /**
     * Get model summary (architecture information)
     * @returns {string} Model summary
     */
    summary() {
        if (!this.isInitialized) {
            return 'Neural network not initialized';
        }
        this.model.summary();
    }
    
    /**
     * Dispose of the model and free memory
     */
    dispose() {
        if (this.model) {
            this.model.dispose();
            this.model = null;
            this.isInitialized = false;
        }
    }
}

