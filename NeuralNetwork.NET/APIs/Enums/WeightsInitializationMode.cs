namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// An <see cref="enum"/> indicating an initialization mode to use for the weights in a network layer
    /// </summary>
    public enum WeightsInitializationMode : byte
    {
        /// <summary>
        /// LeCun uniform distribution, from LeCun 98, Efficient Backprop, <see cref="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf"/>
        /// </summary>
        LeCunUniform,

        /// <summary>
        /// Glorot & Bengio normal distribution, from Glorot & Bengio, AISTATS 2010
        /// </summary>
        GlorotNormal,

        /// <summary>
        /// Gloro & Bengio uniform distribution, <see cref="http://github.com/fchollet/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py"/>
        /// </summary>
        GlorotUniform,

        /// <summary>
        /// He et al. normal distribution, <see cref="http://arxiv.org/abs/1502.01852"/>
        /// </summary>
        HeEtAlNormal,

        /// <summary>
        /// He et al. uniform distribution, <see cref="http://arxiv.org/abs/1502.01852"/>
        /// </summary>
        HeEtAlUniform
    }
}
