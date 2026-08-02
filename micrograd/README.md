More information can be found by looking at the original [micrograd](https://github.com/karpathy/micrograd/tree/master) by Andrej Karpathy

Micrograd is a small autograd engine. It uses backpropagation to find the gradient.

The program uses the Value class which holds a scalar value and its gradient. It can be used in:
mathematical operations including add, multiply, power, exp, and tanh as primitives, each defining their own gradient, while subtract (`a - b` becomes `a + (-b)`) and divide (`a / b` becomes `a * b ** -1`) are made using these.
From this it also allows the construction of more complex mathematical operations.
When calculating the gradient during backpropagation, calling `.backward()` on the final output will go through the whole network in reverse topological order, finding the gradient which is stored in the `.grad` attribute of the Value.

When backpropagating through the network, for each of the primitive operators they have a `_backward` which is used to find the gradient.

An example of defining an operator in the Value class:

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
```

Micrograd can be used to make a neural network.
When making the neural network it also creates a list of the parameters so that they can be adjusted to improve the neural network.

The neural network is the same as an MLP (multi-layer perceptron).
This code makes the neural network:

```python
class Neuron:

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin, ) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

```

Example use for micrograd:
You can define a network by using `n = MLP(3, [4, 4, 1])` which will make a network with four layers: the first has 3 neurons, the second has 4 neurons, the third layer as 4 neurons, and the output layer has 1 neuron.

The input and desired output can be defined as:
```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0] # desired targets
```

To train the network there are three steps: the forward pass, the backward pass, and the update.
The forward pass finds the output based on the input and calculates the loss.
The backward pass finds the gradient for every Value — it also starts by resetting the gradients so they don't accumulate.
The update adjusts the weights and biases based on the gradients calculated in the backward pass.

An example of implementing this is:

```python
for k in range(20):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.1 * p.grad

    print(k, loss.data)
```