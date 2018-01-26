namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// Indicates an activation function to use in a neural network
    /// </summary>
    public enum ActivationType : byte
    {
        /// <summary>
        /// The sigmoid function, 1 / (1 + e^(-x))
        /// </summary>
        Sigmoid,

        /// <summary>
        /// The tanh function, (e^x - e^(-x)) / (e^x + e^(-x))
        /// </summary>
        /// <remarks>It has the advantage of being centered vertically at the origin, instead
        /// of being shifted upwards like the classic sigmoid function</remarks>
        Tanh,

        /// <summary>
        /// A customized tanh function proposed by LeCun (http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
        /// </summary>
        LeCunTanh,

        /// <summary>
        /// The linear rectified function, max(0, x)
        /// </summary>
        /// <remarks>It doesn't saturate like the sigmoid or tanh function and it converges faster</remarks>
        ReLU,

        /// <summary>
        /// The leaky ReLU function, max(0.01x, x)
        /// </summary>
        /// <remarks>It has the advance of having a nonzero gradient for negative values of x, so
        /// a negative neuron won't be stuck there during the rest of the training</remarks>
        LeakyReLU,

        /// <summary>
        /// The absolute linear rectified function, |x|
        /// </summary>
        /// <remarks>It can perform well with images when the dataset contains samples with different brightness levels</remarks>
        AbsoluteReLU,

        /// <summary>
        /// The softmax function, e^x/sum{k}(e^x(k))
        /// </summary>
        Softmax,

        /// <summary>
        /// The softplus function, ln(1 + e^x)
        /// </summary>
        Softplus,

        /// <summary>
        /// The exponential linear unit function, [{ x, x positive}, { e^x - 1, otherwise}];
        /// </summary>
        ELU,

        /// <summary>
        /// A linear activation function that just returns the input value
        /// </summary>
        Identity
    }
}