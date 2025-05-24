# Machine Learning Foundations

# Week - 1

## Types of Models in ML

- Predictive Model
    - Regression Model
    - Classification Model
- Probabilistic Model

## Learning Algorithms

It chooses from a collection of models with same structure but different parameters. Feed data into it and gives the best model.

## Supervised Learning

### Regression

- E.g.: Predict house price from room, area, distance, etc.
- Training data: ${(x^1, y^1), (x^2, y^2),……(x^n, y^n)}$
- $x^i \in R^d, y^i \in R$
- Algorithm outputs a model $f: R^d \to R$
- Loss = $1/n \sum_{i=1}^n (f(x^i) - y^i)^2$
- For a linear model:
    - f(x) = $w^Tx + b = \sum^d_{j=1}w_jx_j +b$

### Classification

- E.g.: Predict if rooms>3 from area & price.
- Training data: ${(x^1, y^1), (x^2, y^2),……(x^n, y^n)}$
- $x^i \in R^d, y^i \in (+1, -1)$
- Algorithm outputs a model $f: R^d \to R$
- Loss = $1/n \sum_{i=1}^n 1(f(x^i) != y^i)$
- For a linear model:
    - $f(x) = sign(w^Tx + b)$

## Types of Data

### Evaluation

For correctly evaluating the models we should use **test data** that is a different subset of the dataset from training data.

### Model Selection

- Learning algorithms just find the “best” model (e.g. with least loss) from a collection of models that human provides them.
- But how do we ensure that the collection that we’re providing the algorithms, is the right collection of models?
    - This is where we use another subset of data called **validation data** which is different from train and test data.

## Unsupervised Learning

- It’s majorly about “understanding data”.
- Data only contain input labels.
- Here, we build models that compress, explain and group data.

### Dimensionality Reduction

- Data: $\{{x^1, x^2, ......, x^n}\}$
- $x^i \in R^d$
- Encoder $f: R^d \to R^{d'}$
- Decoder $g: R^{d'} \to R^d$
- Goal: $g(f(x^i)) \approx x^i$
- Loss = $1/n \sum^n_{i=1} || g(f(x^i)) - x^i||^2$

### Density Estimation

- Data: $\{x^1, x^2, .....,x^n\}$
- $x^i \in R^d$
- Probability mapping $P: R^d \to R_+$ that ‘sums’ to one.
- Goal: P(x) is large if $x\in Data$, and low otherwise.
- Loss = $1/n \sum^n_{i=1} - log(P(x^i))$
- Lower negative log value of P(x) → better model