namespace NeuralNetworkNET.APIs.Enums
{
    /// <summary>
    /// An <see langword="enum"/> indicating an initialization mode to use for the weights in a network layer
    /// </summary>
    public enum WeightsInitializationMode : byte
    {
        /// <summary>
        /// LeCun uniform distribution, from LeCun 98, Efficient Backprop, see <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf">yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf</a>
        /// </summary>
        LeCunUniform,

        /// <summary>
        /// Glorot &amp; Bengio normal distribution, from Glorot &amp; Bengio, AISTATS 2010
        /// </summary>
        GlorotNormal,

        /// <summary>
        /// Glorot &amp; Bengio uniform distribution, see <a href="http://github.com/fchollet/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py">github.com/fchollet/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py</a>
        /// </summary>
        GlorotUniform,

        /// <summary>
        /// He et al. normal distribution, see <a href="http://arxiv.org/abs/1502.01852">arxiv.org/abs/1502.01852</a>
        /// </summary>
        HeEtAlNormal,

        /// <summary>
        /// He et al. uniform distribution, see <a href="http://arxiv.org/abs/1502.01852">arxiv.org/abs/1502.01852</a>
        /// </summary>
        HeEtAlUniform
    }
}
