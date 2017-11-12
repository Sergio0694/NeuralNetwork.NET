namespace NeuralNetworkNET.Networks.Activations.Delegates
{
    /// <summary>
    /// A delegate that represents an activation function (or the derivative of an activation function) used in a neural network
    /// </summary>
    /// <param name="x">The input value</param>
    public delegate float ActivationFunction(float x);
}
