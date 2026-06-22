# Perceptron From Scratch
Perceptron built from first principles: pure Python, no NumPy, no PyTorch.  Trains on logic gates like AND, OR, NAND (linearly separable). Fails on XOR.

## What is a Perceptron?
A perceptron is just a function:
 
```
 y = f(w1*x1 + w2*x2 + ... + wn*xn + bias)
```
 
- `x1, x2, ... xn` — inputs  
- `w1, w2, ... wn` — weights (one per input, controls importance)  
- `bias` — a floating point number (shifts the decision boundary)
- `f` — activation function (step function here)  
- `y` — predicted output (0 or 1)
<br>y is the predicted output. We compare it against the actual output, compute the error, and adjust weights and bias to reduce that error.
That loop is training.

<img width="1123" height="794" alt="image" src="https://github.com/user-attachments/assets/e6c583fe-9d94-4256-8a18-cae6239a1975" /> <br>

A neuron gets signals from many inputs, if the strength of the total signal is strong enough, the neuron fires otherwise it does not.
We micmic this behaviour of the neuron through a perceptron, the perceptron has a similar mechanism. If the sum of all the weighted inputs is more than the threshold, it produces an output of 1 (fires) otherwise 0.

So, what we are doing here is: we are creating a simulated neural tissue. A neuron is trained through repeated stimulus, it strengthens or weakens connections through this training to have a better response next time. Our perceptron would be trained similarly. <br>
<img width="1451" height="622" alt="image" src="https://github.com/user-attachments/assets/4451a496-d757-42ee-aa72-08ac46ee8fda" />
<br>
## How does the perceptron operate
## Weight and Bias Initialization
Before training begins, we initialize weights and bias with random values:
```
weight = [random.random(), random.random()]
bias = random.random()
```
Weight is the importance of an input. In the beginning we don't know which input matters more. So we assign random values at the start, which eventually gets corrected during the training. We don't initialize all the weights with zero because that would create the symmetry problem: if all the weights are zero, every input contributes equally to the weighted sum, so every weight gets the same error, so every weight updates by the same amount. Hence, the network would never learn which input matters more.<br>
The perceptron operates via a straightforward mathematical sequence:
## 1. The Weighted Sum
The perceptron multiplies each input (x) by its corresponding learned weight (w) and adds a bias term (b).
```
z = w1*x1 + w2*x2 + .....+wn*xn +bias
```
Weight is the importance of that input. Each input needs its own dial to control how much it influences the output. Hence, one weight per input.
## 2. The Activation Function
The computed sum is passed into an activation function (typically a step function) to determine the final binary output:
```
y = 1, if z>=0
  = 0, if z<0
```
This gives a binary output for prediction which mimics the output of the neuron: fire or don't fire.

## 3. The Learning Rule
During training, the perceptron compares its predicted output with the actual target output. If it makes a mistake, it updates its weights and bias to improve accuracy for the next pass using the formula:
```
w = w + learning_rate * (actual - predicted) * x
similarly for bias:
bias = bias+ learning_rate * (actual- predcited)
```
Wrong prediction → nudge weights toward correct answer.  
Right prediction → do nothing.  
Repeat for every example, every epoch. That's it.

Learning rate is a small number (like 0.01) that controls how big each weight update step is.
Small Learning rate is slow but stable, big learning rate is fast but brings along the risk of overshooting.
For a classical perceptron (step function), output is 0 or 1 so we can't actually overshoot to the wrong side with a binary input, overshooting is more relevant to continuous activation functions like sigmoid, where bigger jumps can push past the sweet spot.
But we keep the learning rate in the equation for consistency, although it is not a necessity. For our perceptron learning rate just controls the speed.

## Why Does It Only Work on Some Gates?
Because the equation is itself a line:
```
w1*x1 + w2*x2 + bias = 0
```
Perceptron draws this one line and separates one side as 0 and the other side as 1.
Training only moves or rotates this one line. It can never become a curve, it can never bend.
<br>
<b> So for the perceptron to work, the data must be linearly separable: meaning one straight line must be able to separate the two classes. </b>

## Linear Separability
Plot the inputs of AND gate on a 2D graph:
```
(0,0) → 0    (0,1) → 0
(1,0) → 0    (1,1) → 1
```
The only point that outputs 1 is (1,1). You can draw a straight line that puts it on one side and everything else on the other. AND is linearly separable.
<br>
In the plots:<br>
🟢 o : output is 1 (true-fires)<br>
🔴 x : output is 0 (false-silent)
<br>
<table>
  <tr>
    <td><img alt="andgraph" src="https://github.com/user-attachments/assets/acff216d-7c69-4312-bba9-e48dec14a96c" width="400"/></td>
    <td><img alt="anderro" src="https://github.com/user-attachments/assets/ca90be42-a05d-40f9-86ea-4a4b3ce3f154"  width="400"/></td>
  </tr>
</table>
<br>
Similarly for OR and NAND, we can draw a straight line that separates the output classes (true and false).
<br>
<table>
  <tr>
    <td><img width="400" alt="orgraph" src="https://github.com/user-attachments/assets/9767f833-ecf7-4337-acaa-9723f24d4bed" /></td>
    <td><img width="400" alt="orerror" src="https://github.com/user-attachments/assets/27e039c9-5573-443d-b28e-41e9e7dfbf43" /></td>
  </tr>
  <tr>
    <td><img width="400" alt="nandgraph" src="https://github.com/user-attachments/assets/aa24360f-dbab-4091-8c43-382926890ef5" /></td>
    <td><img width="400" alt="nanderror" src="https://github.com/user-attachments/assets/d25ef348-5fdc-4834-8053-f1ba327e244f" /></td>
  </tr>
</table>
<br>

Now plot XOR:
```
(0,0) → 0    (0,1) → 1
(1,0) → 1    (1,1) → 0
```
The 0 output points are (0,0) and (1,1): diagonal corners.

The 1 output points are (0,1) and (1,0): the other diagonal.
If we try drawing one straight line to separate them, we can't. They are interleaved diagonally.<br>
<table>
  <tr>
    <td><img alt="xorgraph" src="https://github.com/user-attachments/assets/5262aa31-6efc-4a77-992a-e38b2cc96877" width="400"/></td>
    <td><img alt="xorerror" src="https://github.com/user-attachments/assets/23177c1c-d127-4783-9180-3ca499177232" width="400"/></td>
  </tr>
</table>
<br>
That is XOR failing exactly as expected.
Error stuck at 4 (all 4 predictions wrong, never converges). Line can't separate them, green and red are diagonally interleaved, no single line can split them.
<br>
To solve XOR, we need two lines minimum. To solve real problems, we need many more.
That's what a Multi-Layer Perceptron (MLP) is: multiple perceptrons stacked in layers. That is what we will build next.
