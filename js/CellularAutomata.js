/**
 * CellularAutomata.js - Coordinates Grid and NeuralNetwork for CA updates
 * Implements synchronous update loop: compute all new states, then apply simultaneously
 */
class CellularAutomata {
    /**
     * Create a new Cellular Automata instance
     * @param {Grid} grid - The grid instance to update
     * @param {NeuralNetwork} neuralNetwork - The neural network for state prediction
     */
    constructor(grid, neuralNetwork) {
        if (!grid || !neuralNetwork) {
            throw new Error('CellularAutomata requires both grid and neuralNetwork');
        }
        
        this.grid = grid;
        this.neuralNetwork = neuralNetwork;
        this.isRunning = false;
        this.animationFrameId = null;
        this.updateCallback = null; // Optional callback after each update
        this.updateInterval = 100; // Milliseconds between updates (default: 100ms = 10 FPS)
        this.maxSteps = null; // Maximum number of steps to run (null = unlimited)
        this.currentStep = 0; // Current step count
        this.isContinuous = false; // If true, ignore maxSteps
        this.completionCallback = null; // Callback called when run completes (due to step limit)
    }
    
    /**
     * Perform a single synchronous update step
     * All cells compute new state from current state, then all update simultaneously
     */
    update() {
        if (!this.neuralNetwork.isInitialized) {
            throw new Error('Neural network not initialized. Call neuralNetwork.initialize() first.');
        }
        
        // Phase 1: Compute all new states from current state
        // We'll collect all inputs first for batch processing (more efficient)
        const inputs = [];
        const cellPositions = [];
        
        for (let y = 0; y < this.grid.height; y++) {
            for (let x = 0; x < this.grid.width; x++) {
                // Get neighbor input (30 values)
                const input = this.grid.getNeighborInput(x, y);
                inputs.push(input);
                cellPositions.push({ x, y });
            }
        }
        
        // Phase 2: Run neural network on all cells (batch prediction for efficiency)
        const newStates = this.neuralNetwork.predictBatch(inputs);
        
        // Phase 3: Apply all new states simultaneously
        for (let i = 0; i < cellPositions.length; i++) {
            const { x, y } = cellPositions[i];
            const newState = newStates[i];
            
            // Update cell with new state
            this.grid.setCell(x, y, newState.on, newState.stateVector);
        }
    }
    
    /**
     * Start continuous update loop
     * @param {Function} callback - Optional callback function called after each update
     * @param {number} interval - Optional update interval in milliseconds (default: 100ms)
     * @param {number} maxSteps - Optional maximum number of steps to run (null = unlimited)
     * @param {boolean} isContinuous - If true, ignore maxSteps and run continuously
     * @param {Function} completionCallback - Optional callback called when run completes (due to step limit)
     */
    start(callback = null, interval = null, maxSteps = null, isContinuous = false, completionCallback = null) {
        if (this.isRunning) {
            console.warn('Cellular Automata is already running');
            return;
        }
        
        this.updateCallback = callback;
        if (interval !== null) {
            this.updateInterval = interval;
        }
        this.maxSteps = maxSteps;
        this.isContinuous = isContinuous;
        this.currentStep = 0;
        this.completionCallback = completionCallback;
        
        this.isRunning = true;
        this._runLoop();
    }
    
    /**
     * Stop continuous update loop
     */
    stop() {
        this.isRunning = false;
        if (this.animationFrameId !== null) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        this.currentStep = 0;
        this.completionCallback = null; // Clear completion callback when manually stopped
    }
    
    /**
     * Internal method: Run the update loop using requestAnimationFrame with timing control
     */
    _runLoop() {
        if (!this.isRunning) {
            return;
        }
        
        const startTime = performance.now();
        
        // Perform update
        try {
            this.update();
            this.currentStep++;
            
            // Check if we've reached the step limit (unless continuous mode)
            if (!this.isContinuous && this.maxSteps !== null && this.currentStep >= this.maxSteps) {
                // Save completion callback before stopping (stop() clears it)
                const completionCallback = this.completionCallback;
                this.stop();
                // Call completion callback if provided
                if (completionCallback) {
                    completionCallback();
                }
                return;
            }
            
            // Call callback if provided
            if (this.updateCallback) {
                this.updateCallback();
            }
        } catch (error) {
            console.error('Error during CA update:', error);
            this.stop();
            return;
        }
        
        // Calculate elapsed time and schedule next update
        const elapsed = performance.now() - startTime;
        const delay = Math.max(0, this.updateInterval - elapsed);
        
        // Use setTimeout for precise timing control
        setTimeout(() => {
            if (this.isRunning) {
                this.animationFrameId = requestAnimationFrame(() => this._runLoop());
            }
        }, delay);
    }
    
    /**
     * Set the update interval (time between updates in milliseconds)
     * @param {number} interval - Milliseconds between updates
     */
    setUpdateInterval(interval) {
        this.updateInterval = Math.max(0, interval);
    }
    
    /**
     * Get current running state
     * @returns {boolean} True if CA is currently running
     */
    getIsRunning() {
        return this.isRunning;
    }
    
    /**
     * Reset the CA to initial state (clears the grid)
     */
    reset() {
        this.stop();
        this.grid.reset();
    }
    
    /**
     * Set a specific cell state manually (useful for initialization)
     * @param {number} x - Column index
     * @param {number} y - Row index
     * @param {boolean} on - On/off state
     * @param {Float32Array} stateVector - Optional state vector (5 floats)
     */
    setCell(x, y, on, stateVector = null) {
        this.grid.setCell(x, y, on, stateVector);
    }
    
    /**
     * Get a cell state
     * @param {number} x - Column index
     * @param {number} y - Row index
     * @returns {Object} Cell state { on: boolean, stateVector: Float32Array(5) }
     */
    getCell(x, y) {
        return this.grid.getCell(x, y);
    }
}

