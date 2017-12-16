using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Structs;

namespace NeuralNetworkNET.DependencyInjection.Delegates
{
    /// <summary>
    /// A delegate that wraps a method that performs the multiplication between two input matrices
    /// </summary>
    /// <param name="m1">The first matrix to multiply</param>
    /// <param name="m2">The second matrix to multiply</param>
    /// <param name="result">The resulting matrix</param>
    public delegate void Multiplication(
        in Tensor m1, in Tensor m2, out Tensor result);

    /// <summary>
    /// A delegate that wraps a method that performs the multiplication between two input matrices and sums a vector to the result
    /// </summary>
    /// <param name="m1">The first matrix to multiply</param>
    /// <param name="m2">The second matrix to multiply</param>
    /// <param name="v">The bias vector to sum</param>
    /// <param name="result">The resulting matrix</param>
    public delegate void MultiplicationWithSum(
        in Tensor m1, float[,] m2, float[] v, out Tensor result);

    /// <summary>
    /// A delegate that wraps a method that executes the given activation function on all the input values
    /// </summary>
    /// <param name="m">The target matrix</param>
    /// <param name="activation">The activation function to use</param>
    /// <param name="result">The resulting matrix</param>
    public delegate void Activation(in Tensor m, [NotNull] ActivationFunction activation, out Tensor result);

    /// <summary>
    /// A delegate that wraps a method that multiplies two matrices, activates the third and performs the product of the two results
    /// </summary>
    /// <param name="m">The matrix to activate</param>
    /// <param name="di">The first matrix to multiply</param>
    /// <param name="wt">The second matrix to multiply</param>
    /// <param name="prime">The activation function to use</param>
    public delegate void MultiplicationAndHadamardProductWithActivation(
        in Tensor m, in Tensor di, in Tensor wt, [NotNull] ActivationFunction prime);
}
