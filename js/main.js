/**
 * main.js - UI event handlers and orchestration for Neural Cellular Automata
 */

// Global references
let grid;
let neuralNetwork;
let cellularAutomata;
let trainer;
let testCanvas;
let testCtx;
let targetCanvas;
let targetCtx;
let targetShape; // 10×10 boolean array for target shape
let isDragging = false; // Track if mouse is being dragged on test canvas
let lastCellX = -1; // Track last modified cell to avoid duplicate toggles
let lastCellY = -1;
let runButton;
let trainButton;
let clearButton;
let genStepsDropdown;
let continuousCheckbox;
let lossValueElement;

/**
 * Initialize the application
 */
function init() {
    // Get canvas elements
    testCanvas = document.getElementById('testCanvas');
    testCtx = testCanvas.getContext('2d');
    targetCanvas = document.getElementById('targetCanvas');
    targetCtx = targetCanvas.getContext('2d');
    
    // Check if canvases are available
    if (!testCanvas || !targetCanvas || !testCtx || !targetCtx) {
        console.error('Failed to initialize canvases');
        return;
    }
    
    // Disable image smoothing for pixel-perfect rendering
    targetCtx.imageSmoothingEnabled = false;
    testCtx.imageSmoothingEnabled = false;
    
    // Initialize grid (9×9 for test canvas - symmetric about center pixel)
    grid = new Grid(9, 9);
    
    // Place a single seed cell at the center of the grid
    const centerX = Math.floor(grid.width / 2);
    const centerY = Math.floor(grid.height / 2);
    grid.setCell(centerX, centerY, true);
    
    // Initialize neural network
    neuralNetwork = new NeuralNetwork({
        hiddenSize1: 16,
        hiddenSize2: 16
    });
    
    // Initialize neural network model
    try {
        neuralNetwork.initialize();
        console.log('Neural network initialized');
    } catch (error) {
        console.error('Failed to initialize neural network:', error);
        alert('Failed to initialize neural network. Please check that TensorFlow.js is loaded.');
        return;
    }
    
    // Initialize cellular automata
    cellularAutomata = new CellularAutomata(grid, neuralNetwork);
    
    // Initialize trainer with GA parameters (will be read from UI)
    trainer = new Trainer(grid, neuralNetwork, cellularAutomata, {
        populationSize: 30,
        mutationRate: 0.15,
        mutationStrength: 0.02,
        eliteCount: 2
    });
    
    // Initialize target shape (5×5 - reduced for memory efficiency)
    targetShape = [];
    for (let y = 0; y < 5; y++) {
        targetShape[y] = [];
        for (let x = 0; x < 5; x++) {
            targetShape[y][x] = false;
        }
    }
    
    // Get button references
    runButton = document.getElementById('runBtn');
    trainButton = document.getElementById('trainBtn');
    clearButton = document.getElementById('clearBtn');
    genStepsDropdown = document.getElementById('genSteps');
    continuousCheckbox = document.getElementById('continuousCheckbox');
    lossValueElement = document.getElementById('lossValue');
    
    // Initial render
    renderTestCanvas();
    renderTargetCanvas();
    
    // Set up target canvas click handler
    targetCanvas.addEventListener('click', handleTargetCanvasClick);
    
    // Set up test canvas click and drag handlers
    testCanvas.addEventListener('mousedown', handleTestCanvasMouseDown);
    testCanvas.addEventListener('mousemove', handleTestCanvasMouseMove);
    testCanvas.addEventListener('mouseup', handleTestCanvasMouseUp);
    testCanvas.addEventListener('mouseleave', handleTestCanvasMouseUp); // Stop dragging if mouse leaves canvas
    
    // Set up button handlers
    document.getElementById('trainBtn').addEventListener('click', handleTrain);
    runButton.addEventListener('click', handleRun);
    clearButton.addEventListener('click', handleClear);
    
    console.log('Initialization complete');
}

/**
 * Render the grid state on the test canvas (9×9)
 */
function renderTestCanvas() {
    // Clear canvas
    testCtx.fillStyle = '#ffffff';
    testCtx.fillRect(0, 0, testCanvas.width, testCanvas.height);
    
    // Draw each cell
    const cellWidth = testCanvas.width / grid.width;
    const cellHeight = testCanvas.height / grid.height;
    
    for (let y = 0; y < grid.height; y++) {
        for (let x = 0; x < grid.width; x++) {
            const cell = grid.getCell(x, y);
            
            // Set color based on on/off state
            testCtx.fillStyle = cell.on ? '#000000' : '#ffffff';
            testCtx.fillRect(
                x * cellWidth,
                y * cellHeight,
                cellWidth,
                cellHeight
            );
        }
    }
}

/**
 * Render the target shape on the target canvas (5×5, scaled to 10×10 canvas)
 */
function renderTargetCanvas() {
    // Clear canvas with white background
    targetCtx.fillStyle = '#ffffff';
    targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
    
    // Draw each cell (scale up 5×5 to 10×10 canvas - 2×2 pixels per cell)
    const cellSize = 2;
    for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
            // Set color based on target shape state
            targetCtx.fillStyle = targetShape[y][x] ? '#000000' : '#ffffff';
            targetCtx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }
    
    // Draw grid lines for better visibility
    targetCtx.strokeStyle = '#e0e0e0';
    targetCtx.lineWidth = 0.1;
    for (let i = 0; i <= 5; i++) {
        targetCtx.beginPath();
        targetCtx.moveTo(i * cellSize, 0);
        targetCtx.lineTo(i * cellSize, 10);
        targetCtx.stroke();
        targetCtx.beginPath();
        targetCtx.moveTo(0, i * cellSize);
        targetCtx.lineTo(10, i * cellSize);
        targetCtx.stroke();
    }
}

/**
 * Handle click on target canvas - toggle pixel state
 */
function handleTargetCanvasClick(event) {
    const rect = targetCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Scale coordinates from CSS size to actual canvas size
    const scaleX = targetCanvas.width / rect.width;
    const scaleY = targetCanvas.height / rect.height;
    const canvasX = x * scaleX;
    const canvasY = y * scaleY;
    
    // Calculate which cell was clicked (5×5 grid, scaled to 10×10 canvas)
    const cellSize = targetCanvas.width / 5;
    
    const cellX = Math.floor(canvasX / cellSize);
    const cellY = Math.floor(canvasY / cellSize);
    
    // Ensure coordinates are within bounds
    if (cellX >= 0 && cellX < 5 && cellY >= 0 && cellY < 5) {
        // Toggle the cell state
        targetShape[cellY][cellX] = !targetShape[cellY][cellX];
        
        // Re-render the target canvas
        renderTargetCanvas();
        
        console.log(`Toggled cell at (${cellX}, ${cellY})`);
    }
}

/**
 * Get the target shape as a 5×5 boolean array
 * @returns {Array} 2D array of booleans
 */
function getTargetShape() {
    return targetShape;
}

/**
 * Calculate and display the loss between current grid state and target shape
 */
function calculateAndDisplayLoss() {
    if (!grid || !targetShape || !lossValueElement) {
        return;
    }
    
    // Check if target shape has any pixels
    let hasTarget = false;
    for (let y = 0; y < 10; y++) {
        for (let x = 0; x < 10; x++) {
            if (targetShape[y][x]) {
                hasTarget = true;
                break;
            }
        }
        if (hasTarget) break;
    }
    
    if (!hasTarget) {
        lossValueElement.textContent = '-';
        return;
    }
    
    // Extract center 5×5 region from 9×9 grid (centered at pixel 4,4)
    const centerX = Math.floor(grid.width / 2) - 2;
    const centerY = Math.floor(grid.height / 2) - 2;
    
    let totalLoss = 0;
    let cellCount = 0;
    let error = 0;
    let totalError = 0;
    
    for (let ty = 0; ty < 5; ty++) {
        for (let tx = 0; tx < 5; tx++) {
            const gridX = centerX + tx;
            const gridY = centerY + ty;
            
            const cell = grid.getCell(gridX, gridY);
            const targetValue = targetShape[ty][tx] ? 1.0 : 0.0;
            const actualValue = cell.on ? 1.0 : 0.0;
            
            // Mean squared error
            error = targetValue - actualValue;
            totalError += error;
            totalLoss += error * error;
            cellCount++;
        }
    }
    console.log("totalError: " + totalError.toFixed(6));
    const loss = totalLoss / cellCount;
    lossValueElement.textContent = loss.toFixed(6) + "    totalError: " + totalError.toFixed(6);
}

/**
 * Handle mousedown on test canvas - start drawing
 */
function handleTestCanvasMouseDown(event) {
    // Pause CA updates while drawing
    if (cellularAutomata && cellularAutomata.getIsRunning()) {
        cellularAutomata.stop();
        runButton.textContent = 'Run';
    }
    
    isDragging = true;
    lastCellX = -1;
    lastCellY = -1;
    toggleTestCanvasCell(event);
}

/**
 * Handle mousemove on test canvas - continue drawing if dragging
 */
function handleTestCanvasMouseMove(event) {
    if (isDragging) {
        toggleTestCanvasCell(event);
    }
}

/**
 * Handle mouseup on test canvas - stop drawing
 */
function handleTestCanvasMouseUp(event) {
    isDragging = false;
    lastCellX = -1;
    lastCellY = -1;
}

/**
 * Toggle a cell on the test canvas at the mouse position
 */
function toggleTestCanvasCell(event) {
    const rect = testCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Scale coordinates from CSS size to actual canvas size
    const scaleX = testCanvas.width / rect.width;
    const scaleY = testCanvas.height / rect.height;
    const canvasX = x * scaleX;
    const canvasY = y * scaleY;
    
    // Calculate which cell was clicked (100×100 grid)
    const cellWidth = testCanvas.width / grid.width;
    const cellHeight = testCanvas.height / grid.height;
    
    const cellX = Math.floor(canvasX / cellWidth);
    const cellY = Math.floor(canvasY / cellHeight);
    
    // Ensure coordinates are within bounds
    if (cellX >= 0 && cellX < grid.width && cellY >= 0 && cellY < grid.height) {
        // Avoid toggling the same cell multiple times during drag
        if (cellX !== lastCellX || cellY !== lastCellY) {
            // Toggle the cell state
            const cell = grid.getCell(cellX, cellY);
            grid.setCell(cellX, cellY, !cell.on);
            
            // Re-render the test canvas
            renderTestCanvas();
            
            // Update last cell position
            lastCellX = cellX;
            lastCellY = cellY;
        }
    }
}

/**
 * Handle Train button click - start training
 */
async function handleTrain() {
    if (!trainer) {
        console.error('Trainer not initialized');
        return;
    }
    
    if (trainer.getIsTraining()) {
        // Stop training
        trainer.stopTraining();
        trainButton.textContent = 'Train';
        trainButton.disabled = false;
        console.log('Training stopped - current best performer is now active');
        
        // The best network is already applied (updated after each generation)
        // Reset grid to seed and render to show current best
        grid.clear();
        const centerX = Math.floor(grid.width / 2);
        const centerY = Math.floor(grid.height / 2);
        grid.setCell(centerX, centerY, true);
        renderTestCanvas();
        calculateAndDisplayLoss();
        
        return;
    }
    
    // Check if target shape has any pixels
    let hasTarget = false;
    for (let y = 0; y < 10; y++) {
        for (let x = 0; x < 10; x++) {
            if (targetShape[y][x]) {
                hasTarget = true;
                break;
            }
        }
        if (hasTarget) break;
    }
    
    if (!hasTarget) {
        alert('Please draw a target shape in the 5×5 editor first!');
        return;
    }
    
    // Stop CA if running
    if (cellularAutomata && cellularAutomata.getIsRunning()) {
        cellularAutomata.stop();
        runButton.textContent = 'Run';
    }
    
    // Start training
    trainButton.textContent = 'Training...';
    trainButton.disabled = true;
    trainButton.textContent = 'Training...';
    console.log('Starting training...');
    
    try {
        // Get number of generations from UI
        const numGenerationsInput = document.getElementById('numGenerations');
        const numGenerations = numGenerationsInput ? parseInt(numGenerationsInput.value, 10) : 100;
        
        // Get genSteps from UI for running the best performer
        const genSteps = genStepsDropdown ? parseInt(genStepsDropdown.value, 10) : 50;
        
        // Train for specified number of generations
        await trainer.train(targetShape, numGenerations, (generation, loss, shouldContinue) => {
            console.log(`Training generation ${generation}/${numGenerations}, loss: ${loss.toFixed(6)}`);
            
            // Update UI to show progress
            if (lossValueElement) {
                lossValueElement.textContent = loss.toFixed(6);
            }
            
            // Run the best performer from this generation on the test grid
            // The best network is already applied to the main network by Trainer
            // Reset grid to seed cell
            grid.clear();
            const centerX = Math.floor(grid.width / 2);
            const centerY = Math.floor(grid.height / 2);
            grid.setCell(centerX, centerY, true);
            
            // Run CA for genSteps to show evolution
            for (let step = 0; step < genSteps; step++) {
                cellularAutomata.update();
            }
            
            // Render the result on test canvas
            renderTestCanvas();
            
            // Calculate and display loss
            calculateAndDisplayLoss();
            
            // Update button text to show progress (button says "Stop" when training)
            trainButton.textContent = `Stop (Gen ${generation}/${numGenerations})`;
            
            return shouldContinue;
        });
        
        console.log('Training completed');
        trainButton.textContent = 'Train';
        trainButton.disabled = false;
        const lossHistory = trainer.getLossHistory();
        if (lossHistory.length > 0) {
            const finalLoss = lossHistory[lossHistory.length - 1];
            console.log(`Final loss: ${finalLoss.toFixed(6)}`);
            if (lossValueElement) {
                lossValueElement.textContent = finalLoss.toFixed(6);
            }
        }
    } catch (error) {
        console.error('Training error:', error);
        alert('Training failed: ' + error.message);
        trainButton.textContent = 'Train';
        trainButton.disabled = false;
    }
}

/**
 * Handle Run button click - start/stop CA update loop
 */
function handleRun() {
    if (!cellularAutomata) {
        console.error('Cellular automata not initialized');
        return;
    }
    
    if (cellularAutomata.getIsRunning()) {
        // Stop the CA
        cellularAutomata.stop();
        runButton.textContent = 'Run';
        calculateAndDisplayLoss();
        console.log('CA stopped');
    } else {
        // Get values from dropdown and checkbox
        const maxSteps = parseInt(genStepsDropdown.value, 10);
        const isContinuous = continuousCheckbox.checked;
        
        // Start the CA with rendering callback
        cellularAutomata.start(() => {
            // Callback after each update: re-render the canvas
            renderTestCanvas();
        }, 100, // 100ms interval (10 FPS)
        isContinuous ? null : maxSteps, // maxSteps (null if continuous)
        isContinuous, // isContinuous flag
        () => {
            // Completion callback: called when step limit is reached
            runButton.textContent = 'Run';
            calculateAndDisplayLoss();
            console.log('CA completed (step limit reached)');
        }
        );
        
        runButton.textContent = 'Stop';
        console.log(`CA started (${isContinuous ? 'continuous' : maxSteps + ' steps'})`);
    }
}

/**
 * Handle Clear button click - reset grid to single seed cell at center
 */
function handleClear() {
    if (!grid) {
        console.error('Grid not initialized');
        return;
    }
    
    // Stop CA if running
    if (cellularAutomata && cellularAutomata.getIsRunning()) {
        cellularAutomata.stop();
        runButton.textContent = 'Run';
    }
    
    // Clear the grid
    grid.clear();
    
    // Place a single seed cell at the center
    const centerX = Math.floor(grid.width / 2);
    const centerY = Math.floor(grid.height / 2);
    grid.setCell(centerX, centerY, true);
    
    // Re-render the test canvas
    renderTestCanvas();
    
    // Update loss display
    calculateAndDisplayLoss();
    
    console.log('Grid cleared and reset to single seed cell');
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

