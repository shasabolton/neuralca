# Backpropagation Through Time (BPTT) Implementation

## Overview

This document explains how Backpropagation Through Time (BPTT) is implemented for training the Neural Cellular Automata. The implementation allows gradients to flow back through multiple CA timesteps, enabling the network to learn temporal dynamics and growth patterns.

## Architecture Changes

### Grid.js - Tensor Conversion

Added two new methods to support tensor-based operations:

- **`toTensor()`**: Converts the current grid state (JavaScript objects) to a TensorFlow tensor of shape `[height, width, 6]`
  - Channel 0: on/off state (0 or 1)
  - Channels 1-5: state vector floats
  
- **`fromTensor(tensor)`**: Updates the grid state from a TensorFlow tensor
  - Used for converting tensor results back to JavaScript objects for rendering

### CellularAutomata.js - Differentiable Updates

Added a new method for differentiable CA updates:

- **`updateTensor(gridTensor)`**: Performs a CA update step using tensors
  - Runs entirely inside TensorFlow's computation graph
  - Handles boundary conditions (padding with zeros)
  - Applies sigmoid to on/off output, tanh to state vector
  - Returns a new tensor representing the updated grid state
  - Uses `tf.tidy()` for automatic memory management

### Trainer.js - Complete Rewrite

Completely rewritten to implement proper BPTT:

#### Key Features

1. **Forward Pass with Loss Tracking**
   - Runs CA forward for N steps (default: 32)
   - Computes loss at multiple timesteps (every N steps, default: every 4 steps)
   - All operations run inside TensorFlow's computation graph

2. **Gradient Computation**
   - Uses `tf.grads()` to compute gradients through the entire forward pass
   - Gradients flow back through all CA timesteps
   - Losses from multiple timesteps are averaged

3. **Weight Updates**
   - Applies gradients to update the shared neural network
   - All cells use the same network, so gradients accumulate across all cells and timesteps

## How BPTT Works

### Training Iteration Flow

```
1. Reset grid to seed cell
2. Convert grid to tensor
3. For each CA step:
   a. Run differentiable CA update (updateTensor)
   b. If loss computation step:
      - Extract center 10×10 region
      - Compare to target shape
      - Store loss tensor
4. Sum/average all step losses
5. Compute gradients through entire computation graph
6. Update network weights
7. Run CA forward again (non-differentiable) to get final state for logging
```

### Why This Works

- **Shared Network**: All cells use the same neural network, so gradients from all cells contribute to weight updates
- **Temporal Learning**: By computing loss at multiple timesteps, the network learns not just the final shape, but how to grow it
- **Gradient Flow**: TensorFlow automatically tracks gradients through all tensor operations, including the CA update steps

### Memory Management

- Uses `tf.tidy()` to automatically clean up intermediate tensors
- TensorFlow keeps tensors alive that are needed for gradient computation
- Final grid state is converted back to JavaScript objects for rendering

## Differences from Previous Implementation

### Old Approach (Removed)
- Only trained on final state after running CA
- No gradient tracking through CA steps
- Loss computed outside computation graph
- Couldn't learn temporal dynamics

### New Approach (BPTT)
- Trains through entire forward pass
- Gradients flow through all timesteps
- Loss computed at multiple timesteps
- Network learns growth patterns, not just final shape
- Enables regeneration/healing behavior

## Configuration Options

- **`caStepsPerIteration`**: Number of CA steps per training iteration (default: 32)
- **`lossEveryNSteps`**: Compute loss every N steps (default: 4)
  - Lower values = more frequent loss computation = better gradient signal but slower
  - Higher values = less frequent = faster but potentially weaker signal

## Performance Considerations

- BPTT requires keeping computation graph for all timesteps
- Memory usage scales with number of timesteps
- For very long sequences, consider truncated BPTT (not implemented)
- Tensor operations are GPU-accelerated when available

## Future Improvements

- Truncated BPTT for very long sequences
- Gradient clipping to prevent exploding gradients
- Learning rate scheduling
- Multiple target shapes (conditional generation)

