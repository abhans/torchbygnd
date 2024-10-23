# PyTorch from the Ground Up

Series on popular deep learning framework `PyTorch` and effort to learn it from the ground up.

As the time goes on, various information will be provided here.

# CH:01 | Linear Regression

`Regression` is type of a task which the goal is to predict a "numerical" value (e.g. price, height, weight).

`Linear regression` is type of a model/approach that is used to capture the "linear correlation" between features of the data.

In order to get reliable results from a `Linear Regression` model, there are several conditions to be met:

1. Correlation between features of data "$\large \mathbf{x}$" and the target $\large y$ must be approx. "LINEAR"

2. Existing `NOISE` in the data $\large \mathbf{X}$ must sampled from "Gaussian Dsitribution"

## Building the Intuition

A `Linear Regression` model can simply be expressed as follows:

$$\large \hat{y} = w_{1}\cdot x_1\ +\ w_{2}\cdot x_2\ + \ldots + w_{d}\cdot x_d + b$$

- $ \hat{y}$: **Predicted** Target/Label
- $d$: Number of features
- $w_{n}$: **Weight** for feature $x_n$
- $b$: **Bias**
- $x_n$: $n^{th}$ **Feature**

If all the weights and features are expressed as one, such as

$$\large \mathbf{x}, \mathbf{w}\in \mathbb {R}^{d}$$

previous expression becomes:

$$\large \hat{y} = \mathbf{w}^\top \mathbf{x} + b$$

- $\mathbf{x}$: **Features** of a "single instance"
- $\mathbf{w}$: **Weights** of a "single instance"

Features can be included in a single matrix called **Design Matrix** such that

$$\large \mathbf{X}\in \mathbb{R}^{n\times d}$$

then the previous expression becomes:

$$\large \hat y = \mathbf{Xw} + b$$

> The goal is to **choose** $\large \mathbf{w}$ and $\large b$ such that $\large \hat y$ is **as close as** to the $\large y$

## Loss Functions

Because of the effects of the `NOISE`, it's almost impossible to predict the correct target values for all possible inputs that represents the problem.
> There will always be an **error** between $\large y$ and $\large \hat{y}$

This error is called `Loss` and it's an important metric to assess the **performance** and the **accuracy** of the model

For regression problems, a common loss is known as `Mean Absolute Error (MAE)` that is:

$$\large l^{(i)}(\mathbf{w}, b) = |\hat y^{(i)}\ - y^{(i)}|
 $$

 - $l^{(i)}(\mathbf w, b)$: **Loss** for a single instance $\large i$
- $\hat{y}^{(i)}$: **Predicted** target
- ${y}^{(i)}$: **Actual/Real** target

In terms of the dataset, the average loss can be expressed as:

$$\large L(\mathbf{w}, b) = \frac{1}{n} \sum_{i = 1}^n l^{(i)}(\mathbf{w}, b) = \frac{1}{n} \sum_{i = 1}^n|\hat y^{(i)}\ - y^{(i)}|$$

- $L(\mathbf{w}, b)$: **Average** Loss
- $n$: Number of instances

## Optimizers

Optimizers are used to iteratively improve the models' parameters at each iteration

This is mostly done with `Gradient Descent` algorithm that works as follows:

1. Randomly initializes the parameters

2. Calculates the gradient of the Loss function with respect to parameters of the model

$$\large \nabla{L(\theta)} =\dfrac{\partial}{\partial\theta_{t}}L(\theta)$$

- $\theta$: Models' parameters
- $t$: Current iteration

3. Updates the parameters by stepping the **opposite direction of the gradient** scaled by **learning rate**

$$\large \theta_{t} \leftarrow \theta_{t} - \mu \cdot \nabla{L(\theta)}$$

- $\mu$: Learning rate

4. Repeat the process for each cycle.

There are various approaches that depends on the changes on the "update" step

> If "update" step;
>
> - Is done over **WHOLE** dataset $\large \rightarrow$ `Batch Gradient Descent`
> 
> - Is done over a **SINGLE RANDOM** instance $\large \rightarrow$ `Stochastic Gradient Descent`
> 
> - Is done over a **BATCH** of instances $\large \rightarrow$ `Minibatch Gradient Descent`

For `Minibatch Gradient Descent`, at iteration $\large t$:

- Sample a minibatch of instances $\large \mathcal{B}_t$ which has $\large |\mathcal{B}|$ of instances. 
- Compute the "gradient" of "average loss" on the "minibatch" with respect to model parameters
- Multiply computed gradient with "learning rate" $\large \eta$

$$\large (\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \frac{1}{|\mathcal{B}|} \sum_{i \in \large \mathcal{B}_t} \partial_{(\mathbf{w},\ b)}\cdot l^{(i)}(\mathbf{w},\ b)$$

- $l^{(i)}(\mathbf{w},\ b)$: **Loss** for an instance $\large i$
- $|\mathcal{B}|$: Number of instances in a **minibatch** (User-defined)
- $\mathcal{B}_t$: Minibatch at iteration $\large t$
- $\eta$: **Learning rate** (User-defined)