/**
 * Grid.js - Manages 100×100 cell states for Neural Cellular Automata
 * Each cell stores: { on: boolean, stateVector: Float32Array(5) }
 */
class Grid {
    constructor(width = 100, height = 100) {
        this.width = width;
        this.height = height;
        this.cells = [];
        
        // Initialize grid with all cells off and zero state vectors
        for (let y = 0; y < height; y++) {
            this.cells[y] = [];
            for (let x = 0; x < width; x++) {
                this.cells[y][x] = {
                    on: false,
                    stateVector: new Float32Array(5) // 5 floats initialized to 0
                };
            }
        }
    }
    
    /**
     * Get cell state at position (x, y)
     * @param {number} x - Column index
     * @param {number} y - Row index
     * @returns {Object} Cell state { on: boolean, stateVector: Float32Array(5) }
     */
    getCell(x, y) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
            // Return boundary cell (off with zero state vector)
            return {
                on: false,
                stateVector: new Float32Array(5)
            };
        }
        return this.cells[y][x];
    }
    
    /**
     * Set cell state at position (x, y)
     * @param {number} x - Column index
     * @param {number} y - Row index
     * @param {boolean} on - On/off state
     * @param {Float32Array} stateVector - Optional state vector (5 floats)
     */
    setCell(x, y, on, stateVector = null) {
        if (x < 0 || x >= this.width || y < 0 || y >= this.height) {
            return; // Out of bounds
        }
        
        this.cells[y][x].on = on;
        if (stateVector !== null && stateVector.length === 5) {
            this.cells[y][x].stateVector.set(stateVector);
        }
    }
    
    /**
     * Get neighbors of a cell (top, bottom, left, right, self)
     * Returns 5 neighbors as specified in the architecture
     * @param {number} x - Column index
     * @param {number} y - Row index
     * @returns {Array} Array of 5 neighbor cell states
     */
    getNeighbors(x, y) {
        return [
            this.getCell(x, y - 1),     // top
            this.getCell(x, y + 1),     // bottom
            this.getCell(x - 1, y),     // left
            this.getCell(x + 1, y),     // right
            this.getCell(x, y)          // self
        ];
    }
    
    /**
     * Get neighbor data as a flat array for neural network input
     * Returns 30 values: 5 neighbors × 6 values (1 on/off + 5 state vector floats)
     * @param {number} x - Column index
     * @param {number} y - Row index
     * @returns {Float32Array} 30-element array for neural network input
     */
    getNeighborInput(x, y) {
        const neighbors = this.getNeighbors(x, y);
        const input = new Float32Array(30);
        
        let idx = 0;
        for (const neighbor of neighbors) {
            // Add on/off state (0 or 1)
            input[idx++] = neighbor.on ? 1.0 : 0.0;
            // Add 5 state vector floats
            for (let i = 0; i < 5; i++) {
                input[idx++] = neighbor.stateVector[i];
            }
        }
        
        return input;
    }
    
    /**
     * Clear all cells (set all to off with zero state vectors)
     */
    clear() {
        for (let y = 0; y < this.height; y++) {
            for (let x = 0; x < this.width; x++) {
                this.cells[y][x].on = false;
                this.cells[y][x].stateVector.fill(0);
            }
        }
    }
    
    /**
     * Reset grid to initial state
     */
    reset() {
        this.clear();
    }
    
    /**
     * Get a copy of the current grid state
     * @returns {Array} 2D array copy of cell states
     */
    getState() {
        const state = [];
        for (let y = 0; y < this.height; y++) {
            state[y] = [];
            for (let x = 0; x < this.width; x++) {
                state[y][x] = {
                    on: this.cells[y][x].on,
                    stateVector: new Float32Array(this.cells[y][x].stateVector)
                };
            }
        }
        return state;
    }
}

