# learning_ai

For the skill section of my Silver DofE I want to learn about AI.
My goal with this project is to learn about what an AI is and how to mbuild a basic AI. 

## Learning Plan
### Phase 1 — Foundations and intuition 

This phase has the goal of learning the basics and to have a proper understanding what a model is and why nural networks work the way they do.

### Phase 2 — Building neural networks from scratch 

Start to code a bask nural network

### Phase 3 — From neural network to GPT
Build a small GPT and use real LLMs through an API.


---

### Week 1: What AI actually is, and what an LLM actually is

Watched the first 3 lessons of [Microsoft’s Generative AI for Beginners](https://github.com/microsoft/generative-ai-for-beginners)

looked at the fundamentals of LLM with the goal of answering the following questions:


#### What is an LLM?

LLMs are a type of AI that is meant to use language like a human.<br>
This is done by predicting text based on probabilities.Text is split into tokens which help the LLM predict the next token (the proses of tokenisation). The tokens are converted into integer to represent them before the LLM uses them. This is how the same prompt can give different responses.

#### What can a LLM do?

LLMs can complete text based tasks like answering questions or writing a short story. It tends to be better with tasks that don't require specific information like writing a short story or summarising text.

#### What can't a LLM do?

Because of pretraining LLMs can't be reliably correct so they can't always be used as a source of information. They are also unable to have emotions as they don't actually think like a human this also mean that they can struggle with tasks that use human common sense. They also can't help with task that it has not be trained for as it is pretrained.

---

### Week 2: How a neural network represents knowledge

I watched ["But what is a neural network?"](https://youtu.be/aircAruvnKk?si=Ua-yuvyIjy_TXIwX) by 3blue1brown, I learned:

Neural networks are made of layers including an input, an output layer and hidden layers in between.

Each layer is made up of neurons that have a bias and an activation. The activation is determined by the activation of all the neurons of the previous layer.

The connection between each neuron have a weight. An activation is determined by the wighted sum (the wights being the wights of the connection) added to the bias, then put into a function(e.g. sigmoid) that results in a value between 0 and 1.

<img src="images/neural_network.png" alt="neural network" width="350" height="269">

I also used the [TensorFlow Playground](https://playground.tensorflow.org) to get an idea of the effect of the weight, how many layers there are, how may neuron are on each layer and other parameters, on how effective the neural network is.
