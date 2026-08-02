more information cna be found by locking at the original [makemore](https://github.com/karpathy/makemore)

i did the first 3 part of make more

make more is ment to make a nume using a training data of ~32000 names contained in [names.txt](names.txt)

### [part 1](makemore.ipynb) a bigram character-level language model:

A character level language model will generate the next character of the name using previously generated characters.

In makemore the training data needs to be changed before it can be used.
Each name is split into individual pairs of characters. there is a symbol added to the beginning and end of each word as well this is done with this code.
An example of this is for `'emma'` it would become:
```
<S>e
em
mm
ma
a<E>
```

This code to do this for the whole dataset and count how many time each bigram will appear:
```python
b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1
```

PyTorch can represent and manipulate this data in a 2d array. This code would do this:

```python
import torch
# create a 27x27 array of integers to represent all the characters.
N = torch.zeros((27, 27), dtype=torch.int32)

# create a sorted set of the unique characters in the training data.
chars = sorted(list(set(''.join(words))))

# create a map of string -> integer for each character.
stoi = {s:i+1 for i,s in enumerate(chars)}

# use `.` to represent beginning and end of each sequence.
stoi['.'] = 0

# create the reverse look up.
itos = {i:s for s,i in stoi.items()}

# populate N with two dimensional mapping of bigrams as integers (ch1/ch2) -> count.
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1
```
When displayed it would look like this:

![Bigram Map](images/bigram-map.png)

With these count the probability of each letter coming after another can be calculated then used in a simple model to predict names:

```python
# prepare a matrix of probabilities
P = (N+1).float()
P /= P.sum(1, keepdims=True) # keepdims=True – important to maintain the dimensions of the matrix.

# using a seeded generator with torch.multinomial (below) to make it deterministic
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
  
    out = []
    ix = 0
    while True:
        p = P[ix]
        
        # use torch.multinomial to take a sample
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
```
This will give:


```
mor.
axx.
minaymoryles.
kondlaisah.
anchshizarie.
```
These are not name but they are vaguely name-like this is due to the issues with a bigram model as it only looks at the previous character.
This results in names like mor as it is common of `m` to start a name and an `m` to be followed by a `o` and a `o` to be followed by an `r` and for an `r` to end the word.
So the bigram say it is a name despite it not being a name.

For this model a negative log likelihood is used to find the lost.

In the program it is done like this:

```python

log_likelihood = 0.0
n = 0

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')
```

For the whole dataset this gives a loss of 2.4543561935424805.


At the moment if there is a pair of characters that appears 0 time the loss is infinite so for example the name `andrejq` would give infinite loss as `jq` appears 0 times.
To fix this the counts are all increase by one so there are no bigrams with a probability of 0.
This does that: `P = (N+1).float()`

The limitation of this method is that when it is used for more than a bigram it grows exponentially. a bigram has 27^2 possibility but a trigram has 27^3 and a ten-gram has 27^10 possible character set,
but the training data doesn't grow meaning that the number of time each character set occurs become close to or 0 make it meaningless this is called the curse of dimensionality – Bengio et al. (2003).
This can be fixed with a neural network


Before, every bigram has been mapped to an integer this can't be used for a neural network so the integers need to be converted into vectors this is done with one hot encoding.

One hot encode has all possible value as floating value within a column that are always 1.0 or 0.0 so every column represent one value.

For example if you only have a value that is A-C A would be 1 0 0 and B would be 0 1 0 and C would be 0 0 1.

As there are 27 characters it would have 27 columns.

This makes it a valid input to a neural network as it has a value for every input neuron.

The bigrams are stored as two values `xs` and `ys` where each is a character in the bigram and when the program runs the `xs` would be the input and the `ys` would be the output.

This is the code to do that:

```python
xs, ys = [], []
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)
```

The network is initialized with random weights and values made with `torch.randn` and it is based on a random distribution.

The training data can be used to apply gradient descent the `.backward()` from PyTorch is used:

```python
import torch.nn.functional as F

for k in range(1000):
  
    # forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # input to the network: one-hot encoding
    logits = xenc @ W # predict log-counts
    counts = logits.exp() # counts, equivalent to N
    probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()
    print(loss.item())
  
    # backward pass
    W.grad = None # set to zero the gradient
    loss.backward()
  
    # update
    W.data += -50 * W.grad
 ```

here `+ 0.01*(W**2).mean()` for model smoothing by pushing the weights towards zero and a uniform distribution.

Now sample can be taken from the neural network
```python
g = torch.Generator().manual_seed(2147483647)

for i in range(5):
    out = []
    ix = 0
    while True:
    
        # ----------
        # BEFORE:
        #p = P[ix]
        # ----------
        # NOW:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W # predict log-counts
        counts = logits.exp() # counts, equivalent to N
        p = counts / counts.sum(1, keepdims=True) # probabilities for next character
        # ----------
        
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
```
This gives
```
mor.
axx.
minaymoryles.
kondlaisah.
anchthizarie.
```
This is very similar to the original model but the advantage of this is as it is taken from the neural network it is more flexible meaning it can be expanded easily, both as it could be used for more than just bigrams and as more layers can be added. 


### [part 2](makemore2.ipynb) multilayer perceptron (MLP) character-level language model:



### [part 3](makemore3.ipynb) inproved (MLP) character-level language model:


