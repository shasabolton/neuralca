/**
 * Game.js - Encapsulates running CA and calculating error
 */
class Game {
    /**
     * Create a new Game instance
     * @param {Grid} grid - The grid instance
     * @param {CellularAutomata} cellularAutomata - The CA instance
     */
    constructor(grid, cellularAutomata) {
        if (!grid || !cellularAutomata) {
            throw new Error('Game requires both grid and cellularAutomata');
        }
        
        this.grid = grid;
        this.cellularAutomata = cellularAutomata;
        this.lossValueElement = null; // UI element for displaying loss (set externally)
    }
    
    /**
     * Set the UI element for displaying loss
     * @param {HTMLElement} element - The element to display loss in
     */
    setLossDisplayElement(element) {
        this.lossValueElement = element;
    }
    
    /**
     * Display a loss value to the UI (simple display without totalError)
     * @param {number} loss - Loss value to display
     */
    displayLoss(loss) {
        if (this.lossValueElement) {
            this.lossValueElement.textContent = loss.toFixed(6);
        }
    }
    
    /**
     * Reset grid to seed cell at center
     */
    resetToSeed() {
        this.grid.clear();
        const centerX = Math.floor(this.grid.width / 2);
        const centerY = Math.floor(this.grid.height / 2);
        this.grid.setCell(centerX, centerY, true);
    }
    
    /**
     * Calculate error between current grid state and target shape
     * Also calculates total error and displays both to UI if lossValueElement is set
     * @param {Array<Array<boolean>>} targetShape - 5×5 boolean array
     * @returns {number} Error value (mean squared error)
     */
    calculateError(targetShape) {
        if (!targetShape || targetShape.length !== 5 || targetShape[0].length !== 5) {
            throw new Error('Target shape must be a 5×5 boolean array');
        }
        
        // Check if target shape has any pixels
        let hasTarget = false;
        for (let y = 0; y < 5; y++) {
            for (let x = 0; x < 5; x++) {
                if (targetShape[y][x]) {
                    hasTarget = true;
                    break;
                }
            }
            if (hasTarget) break;
        }
        
        // If no target pixels and UI element exists, display dash
        if (!hasTarget) {
            if (this.lossValueElement) {
                this.lossValueElement.textContent = '-';
            }
            return 0; // Return 0 loss if no target
        }
        
        // Extract center 5×5 region from 9×9 grid (centered at pixel 4,4)
        const centerX = Math.floor(this.grid.width / 2) - 2;
        const centerY = Math.floor(this.grid.height / 2) - 2;
        
        let totalLoss = 0;
        let totalError = 0;
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
                totalError += error;
                cellCount++;
            }
        }
        
        const loss = totalLoss / cellCount;
        
        // Display to UI if element is set
        if (this.lossValueElement) {
            console.log("totalError: " + totalError.toFixed(6));
            this.lossValueElement.textContent = loss.toFixed(6) + "    totalError: " + totalError.toFixed(6);
        }
        
        return loss;
    }
    
    /**
     * Run CA for specified number of steps, then calculate error
     * @param {number} genSteps - Number of CA steps to run
     * @param {Array<Array<boolean>>} targetShape - 5×5 boolean array for error calculation (optional)
     * @returns {number|null} Error value after running (if targetShape provided), null otherwise
     */
    run(genSteps, targetShape = null) {
        // Run CA for genSteps
        for (let step = 0; step < genSteps; step++) {
            this.cellularAutomata.update();
        }
        
        // Calculate and return error if target shape provided
        if (targetShape) {
            return this.calculateError(targetShape);
        }
        
        return null;
    }
}

