/**
 * main.js - UI event handlers and orchestration for Neural Cellular Automata
 */

// Global references
let grid;
let neuralNetwork;
let cellularAutomata;
let testCanvas;
let testCtx;
let targetCanvas;
let targetCtx;
let targetShape; // 10×10 boolean array for target shape
let isDragging = false; // Track if mouse is being dragged on test canvas
let lastCellX = -1; // Track last modified cell to avoid duplicate toggles
let lastCellY = -1;
let runButton;

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
    
    // Initialize grid
    grid = new Grid(100, 100);
    
    // Initialize neural network
    neuralNetwork = new NeuralNetwork({
        hiddenSize1: 64,
        hiddenSize2: 128
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
    
    // Initialize target shape (10×10)
    targetShape = [];
    for (let y = 0; y < 10; y++) {
        targetShape[y] = [];
        for (let x = 0; x < 10; x++) {
            targetShape[y][x] = false;
        }
    }
    
    // Get button references
    runButton = document.getElementById('runBtn');
    
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
    
    console.log('Initialization complete');
}

/**
 * Render the grid state on the test canvas (100×100)
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
 * Render the target shape on the target canvas (10×10)
 */
function renderTargetCanvas() {
    // Clear canvas with white background
    targetCtx.fillStyle = '#ffffff';
    targetCtx.fillRect(0, 0, targetCanvas.width, targetCanvas.height);
    
    // Draw each pixel (each pixel is 1×1 in a 10×10 canvas)
    for (let y = 0; y < 10; y++) {
        for (let x = 0; x < 10; x++) {
            // Set color based on target shape state
            targetCtx.fillStyle = targetShape[y][x] ? '#000000' : '#ffffff';
            targetCtx.fillRect(x, y, 1, 1);
        }
    }
    
    // Draw grid lines for better visibility (optional, but helpful)
    targetCtx.strokeStyle = '#e0e0e0';
    targetCtx.lineWidth = 0.1;
    for (let i = 0; i <= 10; i++) {
        targetCtx.beginPath();
        targetCtx.moveTo(i, 0);
        targetCtx.lineTo(i, 10);
        targetCtx.stroke();
        targetCtx.beginPath();
        targetCtx.moveTo(0, i);
        targetCtx.lineTo(10, i);
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
    
    // Calculate which pixel was clicked (10×10 grid)
    const pixelWidth = targetCanvas.width / 10;
    const pixelHeight = targetCanvas.height / 10;
    
    const pixelX = Math.floor(canvasX / pixelWidth);
    const pixelY = Math.floor(canvasY / pixelHeight);
    
    // Ensure coordinates are within bounds
    if (pixelX >= 0 && pixelX < 10 && pixelY >= 0 && pixelY < 10) {
        // Toggle the pixel state
        targetShape[pixelY][pixelX] = !targetShape[pixelY][pixelX];
        
        // Re-render the target canvas
        renderTargetCanvas();
        
        console.log(`Toggled pixel at (${pixelX}, ${pixelY})`);
    }
}

/**
 * Get the target shape as a 10×10 boolean array
 * @returns {Array} 2D array of booleans
 */
function getTargetShape() {
    return targetShape;
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
 * Handle Train button click (placeholder)
 */
function handleTrain() {
    console.log('Train button clicked');
    // TODO: Implement training logic
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
        console.log('CA stopped');
    } else {
        // Start the CA with rendering callback
        cellularAutomata.start(() => {
            // Callback after each update: re-render the canvas
            renderTestCanvas();
        }, 100); // 100ms interval (10 FPS)
        
        runButton.textContent = 'Stop';
        console.log('CA started');
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

