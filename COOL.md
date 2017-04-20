<head>
    <title>The Competitive Overcomplete Output Layer</title>
    <!-- CSS -->
    <link rel="stylesheet" href="https://github.com/vogelta/vogelta.github.io/css/main.css">
    <!-- Google fonts -->
    <link href='http://fonts.googleapis.com/css?family=Roboto:400,300' rel='stylesheet' type='text/css'>
    <!-- mathjax -->
    <script type="text/javascript" src="//cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
</head>
# The Competitive Overcomplete Output Layer

[**GitHub**](https://github.com/vogelta/vogelta.github.io)

The widely-used softmax output layer for neural network classifiers is composed of a fully-connected layer with one output for each potential label, followed by the [softmax function](https://en.wikipedia.org/wiki/Softmax_function). The properties that make this layer a default choice include that it has a simple derivative and that it produces nonnegative scores that sum to one - a probability distribution over the mutually-exclusive output labels - meaning that the network can be trained by minimizing the [cross-entropy loss](https://en.wikipedia.org/wiki/Cross_entropy). The Competitive Overcomplete Output Layer introduced in [Kardan and Stanley](https://arxiv.org/abs/1609.02226), also has those appealing properties - except that its scores have a sum less-than-or-equal-to one, meaning they technically don't define a probability distribution. This doesn't prevent the use of the cross-entropy loss<a href="#note1" id="note1ref"><sup>1</sup></a>, however, so a network using a softmax output layer can still work when using the COOL instead. But why would the COOL be used instead, when it requires extra parameters and extra computation<a href="#note2" id="note2ref"><sup>2</sup></a>? Despite those extra parameters, the COOL actually acts as a [regularizer](https://en.wikipedia.org/wiki/Regularization_(mathematics)) - and moreover, one with an interesting action which (when using rectifier nonlinearities) can be used to define a final model that is not only more accurate, but also smaller than the equivalent softmax network.

The idea behind the COOL is that, instead of there being just one unit for each label in the network's final fully-connected layer, each label can have multiple units corresponding to it - hence, 'Overcomplete' (how many of these units there are for some label is termed its 'Degree-Of-Overcompleteness'). The softmax function is then applied across all of these units ('Competitive'), just as it is in the standard layer. The resulting scores representing each label are then combined into a single value by taking their geometric mean, or alternatively, their minimum<a href="#note3" id="note3ref"><sup>3</sup></a>. Finally, each of these means is multiplied by the corresponding degree-of-overcompleteness, to compensate for its share of the post-softmax distribution having been spread across that many units.

---
![The COOL](https://github.com/vogelta/vogelta.github.io/raw/master/assets/COOL_Figure.png)
 
*Adapted from [Kardan and Stanley](https://arxiv.org/abs/1609.02226). On the left, the classic softmax layer, and on the right, the COOL. The dots are the units of the fully-connected layer, and the blue box represents the softmax function. In the COOL, there are multiple units for each label (grouped in yellow boxes) which are combined after the softmax operation into a single final score.*

---

A critical point to note is that the geometric mean (and the minimum) is maximized for a given sum when all inputs are equal. Thus, at the point directly after the softmax function, a flawless classifier should assign zero scores to all units except to the \\(M\\) units representing the true label, which score \\(\frac{1}{M}\\) each. The geometric mean is then also \\(\frac{1}{M}\\), which gets multiplied by \\(M\\) for a perfect final output of 1. This point also reveals the reason why the arithmetic mean cannot be used instead: it does not encourage these outputs to be equal, which ruins any benefit over the standard softmax.

---

**_Gradients on the Output-Layer Units_**

*\\(z_i\\) is a unit of the fully-connected layer, with a value of \\(y_i\\) after the softmax operation*

| Layer   | Gradient |
| ------- | -------- |
| Softmax | \\(\displaystyle \frac{\partial L}{\partial z_i} = y_i-1\\) if \\(z_i\\) represents the true label, else \\(y_i\\) |
| COOL    | \\(\displaystyle \frac{\partial L}{\partial z_i} = y_i-\frac{1}{M}\\) if \\(z_i\\) is one of the \\(M\\) units representing the true label, else \\(y_i\\) |
| MinCOOL | \\(\displaystyle \frac{\partial L}{\partial z_i} = y_i-1\\) if \\(z_i\\) is the lowest-scoring of the units representing the true label, else \\(y_i\\) |

---

No matter their present value, the gradients on the COOL's 'true' units in the fully-connected layer (that is, prior to the softmax operation) push them toward taking a \\(\frac{1}{M}\\) value after the softmax. Therefore to call the units 'competitive' could be slightly misleading: the anthropomorphized units' motivations are more clearly defined as only trying to take their \\(\frac{1}{M}\\) share, and no more. In the gradient expressions it can also be seen that the using-minimum-instead-of-geometric-mean MinCOOL never has all-zero gradients - in fact, the magnitudes of its gradients always sum to 1.

Clearly, this scheme introduces extra parameters and computation over the softmax layer. To be precise, the number of parameters in the output layer are increased by a factor of the average degree-of-overcompleteness - five units for each label means five times as many parameters, as each unit gets its own set of weights. The number of exponentiations and divisions required for the softmax function are increased similarly. In some applications (for example, in Natural Language Processing where the output may be one of 10,000 or more words) the standard softmax is already impractical to use: in such cases the COOL would not be appropriate. On the other hand, when the output layer is a relatively small part of the overall network then this disadvantage also becomes relatively smaller.

---
![Loss on MNIST](https://github.com/vogelta/vogelta.github.io/raw/master/assets/MNIST_Loss.png) ![Accuracy on CIFAR-10](https://github.com/vogelta/vogelta.github.io/raw/master/assets/CIFAR_Accuracy.png) 

*In experiments with similar networks trained using COOL and softmax outputs, the COOL clearly performed better (MinCOOL refers to a COOL where the minimum is used in place of geometric mean). The COOL and MinCOOL took about 5% / 15% per batch longer to train than softmax, respectively. The original paper also reported improved accuracy on CIFAR-100 with the COOL.* Datasets: [MNIST](http://yann.lecun.com/exdb/mnist/) / [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html)

---

## The COOL as a Regularizer

When considering COOL as a regularizer compared to a softmax output layer, it might seem intuitive that because the only differences in network architecture are at the final layer, that would be where the regularization effect occurs. But as transfer-learning experiments show, most of the improvement in fact comes from finding better features at the earlier layers: training a softmax output using pretrained COOL features achieves a result comparable to the COOL network, and vice versa a COOL output on the softmax features performs comparably to the original softmax network.

---
**_Transfer-Learning Results on CIFAR-10_**

|     | Softmax Features | COOL Features | MinCOOL Features |
| :----------------: | :-----: | :-----: | :-----: |
| **Softmax Output** | 0.6992  | 0.7381  | 0.7472  |
| **COOL Output**    | 0.7041  | 0.7413  | 0.7453  |
| **MinCOOL Output** | 0.7022  | 0.7386  | 0.7485  |

---

During training, the COOL's units learn to return closely-matched outputs (in order to score \\(\frac{1}{M}\\) when true, and 0 when false). Assuming the training set is representative enough that the outputs are also similar on a new example taken from the same distribution, swapping \\(M\\) different-but-very-similar units for \\(M\\) copies of the same unit does not change the network's behaviour much - and a COOL with identical units is equivalent to a softmax layer using the same fully-connected layer weights for each label (with a correction term added to the biases). The layers share an objective - minimizing cross-entropy loss - meaning a COOL that has become approximately like a softmax layer is also drawn to a softmax-layer local minimum<a href="#note4" id="note4ref"><sup>4</sup></a>, so (as observed) the two layers are expected to produce similar performance when given the same features.

In order to explain the regularization and transfer-learning results, it must be that the COOL causes the earlier stages of the network to learn better features. However, when using a rectifier activation function (and NOT using [batch normalization](https://arxiv.org/abs/1502.03167)), at the outputs of intermediate layers there are in fact many more ['dead units'](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Potential_problems) - units that are always 0, no matter the input - which would usually be taken as a sign of worse performance. The dead units do not carry any information about the input, so some of the potential capacity of the network is wasted. Dead units are most commonly encountered as a result of too-high learning rates, which lead to large updates that irrevocably 'kill' the unit. However, the dead units still appeared with COOL even when using lower learning rates and after gradient clipping.

---
![Dead units on CIFAR during training](https://github.com/vogelta/vogelta.github.io/raw/master/assets/CIFAR_Dead.png) ![Activity of MNIST units](https://github.com/vogelta/vogelta.github.io/raw/master/assets/MNIST_Activity.png)

*The dead-unit effect is stronger for MinCOOL (minimum) than regular COOL (geometric mean). As well as the completely-dead units, there is also a 'tail' of many units that are very rarely activated: on the MNIST dataset, 102 (out of 1024) of the MinCOOL network's penultimate-layer units are greater than zero on between one and ten (out of 55000) training-set examples.*

---

### The 'Dead Units'

So where are the dead units coming from? To gain an intuition, consider just two examples of the same ground-truth label, whose representations at some layer in the main body of the network are exactly the same as each other, except at a single entry: say one of the examples has this unit at zero, and the other is at one. For simplicity, assume that on both of these examples, all of the COOL's units corresponding to labels other than the ground-truth score zero after the softmax operation.

As above, the COOL's 'true' outputs should be as equal as possible in order to minimize loss. With this in mind, a useful tool to have is the concept of a 'most-balanced' value for this unit: given the matching values in the rest of the layer, and given the fixed weights connecting that layer to the outputs of the COOL, what value should the unit in question take to result in as-balanced-as-possible 'true' outputs? This would ideally be a value that would result in perfectly-equal true outputs, but as the degree-of-overcompleteness increases, it becomes more likely that it is not possible to have them all exactly balanced.

At the penultimate layer - the one just before the COOL - the 'most-balanced' value usually lies at the minimum of an approximately parabola-shaped curve of loss-versus-value (except in rarely-seen-in-practice situations<a href="#note5" id="note5ref"><sup>5</sup></a>). Going further back into the network makes this curve less smooth, and may introduce local minima, but as long as the unit's value will be decreasing on examples where it is above the 'most-balanced' value, the same argument can be applied.

Through the gradients on the weights connecting to the 'true' outputs (and also the gradients on the units in the rest of the layer), the 'most-balanced' value is updated toward the value that the unit took in the example being used at the current training step. Assume that the rest of the layer does not change - or that it changes equally on the examples in question - such that they still share the same 'most-balanced' value: by moving toward each of the 0-valued and 1-valued examples in turn, it would be expected to end up at 0.5, the average.

At the same time, for each example, the gradients propagated back to the unequal unit move it in direction of the 'most-balanced' value. With the 'most-balanced' value at 0.5, the example at 0 would push this unit upwards, and the example at 1 would push it downward<a href="#note6" id="note6ref"><sup>6</sup></a>. However, these gradients are not directly part of the eventual update - rather, they are passed through the network in order to find the gradients on the weights at each layer. As these gradients are passed back through the network, they are scaled in proportion to the slope that the nonlinear activation function applied to the corresponding unit: with a [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) the gradient is blocked - multiplied by zero - when the output of the function was zero for that example. When the output had a positive value, the slope was one, and so the gradient is unchanged. 

Because of the nonlinearity, the 'increase' gradient from the 0-valued example is blocked, but the 'decrease' from the 1-valued one is not. So, say the higher one is reduced, to 0.5: on the subsequent iterations, the 'most-balanced' value will again move toward the average, which is now at 0.25, which causes the higher example to be reduced further. This same cycle can repeat again and again, until both examples have this unit at zero. 

On a larger (and less simplified) scale, this basic process can cause a unit which is only in the lower-slope region of the activation function<a href="#note7" id="note7ref"><sup>7</sup></a> on a few examples initially, to eventually be in that zone - at zero, when using the ReLU - on every single one: a dead unit. 

### Why does the MinCOOL make more 'Dead Units'?

Another thing to note in the way the dead units form, is that significantly more of them are created when using the minimum to combine the values of the COOL's units. As in the geometric-mean version, the best possible solution would return zero on all 'false' outputs in the COOL, and have equal 'true' outputs. However, unlike with the geometric mean, the gradients do not point toward this solution: at each step, they are solely in the direction of increasing the value of whichever of the 'true' units was the lowest. In most cases, increasing the lowest of the true outputs - at least, up to the point that it is no longer the lowest - is also increasing the overall geometric mean. Re-using the same 'actual' and 'most-balanced' values concept, the MinCOOL gradients will generally be pointing in the same directions as in the regular COOL.

The difference is that rather than having a magnitude in proportion to how far apart those values are, the gradient is pointing the lowest-true-output toward \\(\frac{1}{M}\\) - and then beyond, all the way up to 1. With the gradients all becoming much larger, the resulting bias due to the nonlinearity is also larger. Under the assumptions used, this doesn't actually make a difference: the same larger updates could be achieved by raising the learning rate. Of course, in practice there are also gradients decreasing the never-actually-zero 'false' outputs and increasing the 'true' ones: the MinCOOL's dead-unit-making bias is larger relative to this signal than the geometric-mean COOL's is, so it can be expected to have a stronger effect over the course of training the network.

### The Real Regularizer 

Having justified that they can be treated as something other than a flaw caused by poor hyperparameter settings, it is tempting to credit the COOL's effect as a regularizer to the dead units. 
After all, it is well-known that neural networks are generally [overparametrized](https://arxiv.org/abs/1306.0543): even if they don't show it by having dead units, networks trained with the softmax output layer are not using their maximum capacity either. Because of this, reducing the network in a principled way - such as killing off some of the less useful units - can help to stop the network overfitting the training set, without harming its overall performance. Most similarly, this was demonstrated in [Network Trimming](https://arxiv.org/abs/1607.03250) by Hu et al: iteratively dropping the units returning the most zeros (after a ReLU activation) from the network, then continuing training. Under this scheme, performance was improved while shrinking the network.

Despite this, other experiments demonstrate that although they might be helpful, the dead units are not the primary means by which the COOL improves performance. When using a nonlinearity that does not allow the creation of dead units, such as the [Exponential Linear Unit](https://arxiv.org/abs/1511.07289), or when using [batch normalization](https://arxiv.org/abs/1502.03167) (which prevents dead units by, as the name suggests, normalization), the COOL maintains its performance advantage over an identical network using the softmax output. 

To see why, start by noting that although the weights do usually end up becoming more and more similar during training, the obvious way of achieving balanced outputs - equal weight vectors for all of the COOL's units for that label - is not given any special significance by the COOL's gradients. Instead, any solution returning equal 'true' outputs on all examples of the training set is treated as equally as good.
Such a solution can be identified label-by-label, by taking the weight vectors (with biases appended) belonging to the COOL's units for that label, and subtracting the first of these vectors from each of the others to define the rows of a matrix. If all the training-set examples with that ground-truth label have penultimate-layer representations which lie in the [null space](https://en.wikipedia.org/wiki/Kernel_(linear_algebra)) of this matrix, then they all result in equal true outputs<a href="#note8" id="note8ref"><sup>8</sup></a>.

By moving the penultimate-layer representations of the training-set examples toward different null spaces, they become better separated by label, which improves their generalization performance. Also, these null spaces make it harder for the network to overfit to the training set: imagine an outlier example of label \\(A\\), which is placed in a region that truthfully should be classified as label \\(B\\). Realigning the null space of \\(A\\) closer to this example would make things worse for the rest of the \\(A\\) examples, so instead, the network must assign more of the post-softmax distribution to the \\(A\\) outputs in order to increase its score. If this outlier is closer the null space of \\(B\\) - such that \\(B\\)'s outputs are significantly more balanced than \\(A\\)'s - then \\(A\\)'s units need to take a significantly higher proportion of the total softmax output than they would have needed otherwise, for this example to be classified as \\(A\\). With a higher bar set against outliers, the COOL's test-set performance is improved.

## Network Compression

As the Network Trimming paper also showed, making the network smaller represents a goal in itself. Because the dead units always return zero, they can safely be removed from the network without changing the results: the network can ignore all computation associated to them, and all the weights going into and out of them can be forgotten. The biggest benefits can usually be found in shrinking the fully-connected layers near the final stages of the network - they usually contain the most weights - and indeed, this is where most of the COOL's dead units formed in practice. Because the same weights are used at every position for the filters in a convolutional layer, every one of those positions needs to be 'dead' before those weights can be forgotten without any risk. Also, as mentioned previously, the COOL's units for each label learn to produce almost the same outputs - which means by taking the weights of just one unit for each class and forgetting the rest (and using those weights in a normal softmax layer), the final trained COOL layer itself can be made smaller without greatly impacting performance.

---

*These tables show for each network the numbers of dead units (on the training set) created, how many weights could be forgotten as a result of removing those units, and the test-set accuracy. The COOL networks are presented as having been changed to use a softmax layer (using just the first of its trained units for each label), so the total network size is the same as in the softmax case.*

*Layer 8 of the CIFAR networks and Layer 5 of the MNIST networks are the layers just before the output layer. Layer 2 in the MNIST networks is the second convolutional layer. No other layers contained weights that were able to be safely forgotten.*

**_CIFAR_**

| Output Layer | Layer-8 Dead Units | Weights Forgotten | % Reduction | Accuracy |
| ------- | ---: | ------: | -----: | -----: |
| Total   | 512  | 3230778 | -      | -      |
| **Softmax** | 10   | 62440   | 1.93%  | 0.7020 |
| **COOL**    | 35   | 215425  | 6.67%  | 0.7396 |
| **MinCOOL** | 161  | 990995  | 30.67% | 0.7457 |

**_MNIST_**

| Output Layer | Layer-5 Dead Units | Layer-2 Dead Channels | Weights Forgotten | % Reduction | Accuracy |
| ------- | ---: | ---: | ------: | -----: | -----: |
| Total   | 1024 | 64   | 3274634 | -      | -      |
| **Softmax** | 5    | 0    | 15735   | 0.48%  | 0.9930 |
| **COOL**    | 166  | 0    | 522402  | 15.95% | 0.9942 |
| **MinCOOL** | 407  | 10   | 1591169 | 48.59% | 0.9938 |

---

Aside from requiring the ReLU, and preventing the use of batch normalization, another significant disadvantage of the COOL as a model-compression technique is that there are only rough ways to control the final model size - by changing the initial size, and perhaps the degree-of-overcompleteness - and it is difficult to guess ahead of time how large the model will end up. Furthermore, in this domain the COOL does not compare in effectiveness to techniques designed with compression as an explicit goal. On the other hand, given a standard softmax network, applying the COOL is very simple - no other changes to the architecture or training setup are needed - and introduces few new hyperparameters.

## Is It Useful?

The COOL's performance as a regularizer - both in the examples described here, and on the CIFAR-100 dataset in the original paper - did not approach state-of-the-art levels, although this was to be expected given the relatively simple networks it was applied atop of. Combining the COOL with other techniques such as [data augmentation](https://arxiv.org/abs/1609.08764) and applying it to more powerful networks would be expected to achieve more competitive results.

In the end, there are plenty of other methods for regularization and plenty of other methods for model-compression, many of which are compatible with each other. Being able to do both at once is convenient, but not inherently better than a combination of approaches. Where the COOL shone is as a challenge to some at-first-glance logical assumptions in softmax layers and neural network training: allowing a less-than-one sum of label probabilities led to better scores overall, and dead units were a sign of better-performing features - and it has to be said that moving from one- to multiple-units-per-concept (although they are still combined into one final score) is particularly appealing for those who like their machine learning to come with (tenuous) neuroscience analogies. 


---


*<a id="note1" href="#note1ref"><sup>1</sup></a> Minimizing cross-entropy loss is equivalent to maximizing the probability assigned to the true class for each example, so having the 'probabilities' sum to less than one is valid - it simply represents a handicap against achieving low loss. Scaling up the scores to sum to one might seem like a natural thing to do, except that it ruins the regularizer effect of the COOL - so they should not be scaled during training, only during testing/inference.*

*<a id="note2" href="#note2ref"><sup>2</sup></a> In the [original paper](https://arxiv.org/abs/1609.02226), the authors suggested that the lower sum-of-scores could combat 'overgeneralization', as it removes the assumption that any input (including, for example, random noise) must lie somewhere amongst the output labels. However, much as would happen if a 'none-of-the-above' output was added without changing the dataset, the network learns to avoid as strongly as possible the option to give out lower scores - the COOL network trained on the MNIST dataset returned scores totalling more than 0.995 on random inputs.*

*<a id="note3" href="#note3ref"><sup>3</sup></a> The geometric mean and minimum are the most appealing options for combining the COOL's units into a single score, because they have simple gradients (unlike the forms of the [generalized mean](https://en.wikipedia.org/wiki/Generalized_mean) that lie between these two), while also preserving the network's ability - in theory - to express any final probability distribution over the labels: simply taking the product of the units, for example, is unable to return final scores of 0.5 on two labels at the same time.*

*<a id="note4" href="#note4ref"><sup>4</sup></a> It may be that the COOL and the softmax layer are biased toward different softmax local minima, given the way training the COOL involves reaching a 'consensus' between the random starting initializations of the units within each label. But it is not likely to make a difference: theory and empirical [evidence](https://arxiv.org/abs/1412.0233) suggest that most local minima of large multilayer neural networks have similar test-set performance.*

*<a id="note5" href="#note5ref"><sup>5</sup></a> For example, when all the weights connecting the unit in question to the true-label COOL outputs are equal: changing the unit's value would have no effect on loss (under the assumption that the 'false' units are already scoring zero), so the curve of loss-versus-value is flat.*

*<a id="note6" href="#note6ref"><sup>6</sup></a> This is unlike what happens using the softmax layer, where these examples (having the same label) will point in the same direction: both upward, if increasing the unit favours the true output more than the false ones, or both downward, if it does not.*

*<a id="note7" href="#note7ref"><sup>7</sup></a> Training a COOL using the [LeakyReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Leaky_ReLUs) activation created almost as many of its 'dead' (always below zero) and 'nearly-dead' (above zero on very few examples) units as a ReLU version. With the [ELU](https://arxiv.org/abs/1511.07289) activation, the slope gradually decreases for inputs below zero. Training a COOL network using this function created units in varying stages of 'dying': some never took a value above -0.9 (the minimum possible value being -1), while others had their average value somewhere closer to zero.*

*<a id="note8" href="#note8ref"><sup>8</sup></a> In practice, the weight vectors will not be so finely balanced as to be linear combinations of each other, and so the dimension of this null space is \\(N-M+2\\), where \\(N\\) is the number of units in the penultimate layer (i.e. not counting the biases), and \\(M\\) is the degree-of-overcompleteness for that label.*


