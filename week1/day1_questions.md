Question 1 — 
You said neurons learn patterns and we update weights to minimise the loss. How does the update actually happen? Walk me through what happens mathematically after you calculate the loss. Don't just say "backpropagation" — explain what backpropagation is actually doing step by step.

Answer 1 - 
Suppose we have 100 data points and we make predictions for 100 of them. Then we will calculate the gap between predicted and actual value. Once we have this gap for all 100 data points, then we will calculate MSE. This will be our loss. Now we will calculate the derivative of loss w.r.t weight and bias at each neuron . This will be the gradient which will tell how the loss will change when we update the weights and bias accordingly. Once we have the gradient then based on the learning rate defined, all neurons will update their weights and bias accordingly.
calculating the derivative of loss w.r.t weight and bias at each neuron is right.but how does it get from the output layer all the way back to the first hidden layer? That's the chain rule in action — the gradient at each layer depends on the gradient of the layer ahead of it. That's why it's called backpropagation — it flows backwards layer by layer, multiplying gradients using the chain rule.

Question 2 — 
You mentioned the Universal Approximation Theorem. In your own words — what does it actually guarantee, and what does it not guarantee? Why does this matter practically?

Answer 2- this theorem guarantee that if we have enough hidden layers and enough neurons within each layer , then any pattern in this whole world can be learnt by the neural network and represented as a mathematical notation . This matters because if the architecture is not dense enough to understand the pattern, then it will lead to underfitting and if it is too dense then it might learn the noise as well. So it is very important to design our ANN such that it can balance between the two.

what it does NOT guarantee":
The theorem guarantees that a network can represent any function — it says nothing about whether your training process will actually find that representation. It doesn't guarantee:

That you have enough data to learn the pattern
That gradient descent will converge to the right solution
That it will generalise to unseen data
How long training will take


Question 3 — 

You listed the steps to solve an ML problem correctly. But you went straight from "Define ANN" to "Train ANN." What's missing between those two steps that would cause your training to completely fail if skipped?

Answer 3 - 
Normalisation / Feature Scaling.

Built in normalization techniques -
Min-Max Scaling    → squish everything to 0-1
                   → use when you know the bounds of your data

Standardisation    → mean=0, std=1 (Z-score)
(most common)      → use when data is roughly gaussian
                   → what sklearn's StandardScaler does

Batch Normalisation → normalise activations between layers
                    → built into deep networks

Question 4 -
Training loop in plain English pseudocode — not Python, just words:

Answer 4 -
For each batch:
  1. Forward pass     → get predictions
  2. Calculate loss   → measure how wrong we are
  3. Zero gradients   → clear previous batch's gradients
  4. Backward pass    → calculate new gradients
  5. Gradient descent → update weights and biases
