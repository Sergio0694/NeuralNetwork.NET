using System;
using System.IO;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Cost;
using NeuralNetworkNET.Networks.Cost.Delegates;
using NeuralNetworkNET.Networks.Layers.Cpu;
using Newtonsoft.Json;

namespace NeuralNetworkNET.Networks.Layers.Abstract
{
    /// <summary>
    /// The base class for all the output network layers
    /// </summary>
    [JsonObject(MemberSerialization.OptIn)]
    internal abstract class OutputLayerBase : FullyConnectedLayer
    {
        #region Fields and parameters

        /// <summary>
        /// Gets the cost function for the current layer
        /// </summary>
        [JsonProperty(nameof(CostFunctionType), Order = 6)]
        public CostFunctionType CostFunctionType { get; }

        /// <summary>
        /// Gets the cost function implementations used in the current layer
        /// </summary>
        public (CostFunction Cost, CostFunctionPrime CostPrime) CostFunctions { get; }

        #endregion

        protected OutputLayerBase(in TensorInfo input, int outputs, ActivationType activation, CostFunctionType cost, WeightsInitializationMode weightsMode, BiasInitializationMode biasMode)
            : base(input, outputs, activation, weightsMode, biasMode)
        {
            CostFunctionType = cost;
            CostFunctions = CostFunctionProvider.GetCostFunctions(cost);
        }

        protected OutputLayerBase(in TensorInfo input, int outputs, [NotNull] float[] weights, [NotNull] float[] biases, ActivationType activation, CostFunctionType cost)
            : base(input, outputs, weights, biases, activation)
        {
            CostFunctionType = cost;
            CostFunctions = CostFunctionProvider.GetCostFunctions(cost);
        }

        
        #pragma warning disable CS0809 // Backpropagation method replaced with overload with expected output values

        [Obsolete("Invalid for an output layer, use the overload with the output tensor instead", true)]
        public override void Backpropagate(in Tensor x, in Tensor y, in Tensor dy, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
            => throw new InvalidOperationException();

        #pragma warning restore CS0809

        /// <summary>
        /// Computes the output delta, with respect to the cost function of the network
        /// </summary>
        /// <param name="x">The layer inputs in the forward pass</param>
        /// <param name="yHat">The estimated outputs for the network</param>
        /// <param name="y">The expected outputs for the used inputs</param>
        /// <param name="z">The activity on the output layer</param>
        /// <param name="dx">The backpropagated error</param>
        /// <param name="dJdw">The resulting gradient with respect to the weights</param>
        /// <param name="dJdb">The resulting gradient with respect to the biases</param>
        public virtual unsafe void Backpropagate(in Tensor x, in Tensor yHat, in Tensor y, in Tensor z, in Tensor dx, out Tensor dJdw, out Tensor dJdb)
        {
            Tensor.Like(y, out Tensor dy);
            CostFunctions.CostPrime(yHat, y, z, ActivationFunctions.ActivationPrime, dy);
            fixed (float* pw = Weights)
            {
                Tensor.Reshape(pw, InputInfo.Size, OutputInfo.Size, out Tensor w);
                CpuDnn.FullyConnectedBackwardData(w, dy, dx);
            }
            Tensor.New(InputInfo.Size, OutputInfo.Size, out Tensor dw);
            CpuDnn.FullyConnectedBackwardFilter(x, dy, dw);
            dw.Reshape(1, dw.Size, out dJdw); // Flatten the result
            Tensor.New(1, Biases.Length, out dJdb);
            CpuDnn.FullyConnectedBackwardBias(dy, dJdb);
            dy.Free();
        }

        /// <summary>
        /// Calculates the output cost with respect to the cost function currently in use
        /// </summary>
        /// <param name="yHat">The estimated output for the network</param>
        /// <param name="y">The Expected outputs for the inputs used</param>
        [Pure]
        [CollectionAccess(CollectionAccessType.Read)]
        public float CalculateCost(in Tensor yHat, in Tensor y) => CostFunctions.Cost(yHat, y);

        #region Equality check

        /// <inheritdoc/>
        public override bool Equals(INetworkLayer other)
        {
            if (!base.Equals(other)) return false;
            return other is OutputLayerBase layer &&
                   CostFunctionType == layer.CostFunctionType;
        }

        #endregion

        /// <inheritdoc/>
        public override void Serialize(Stream stream)
        {
            base.Serialize(stream);
            stream.Write(CostFunctionType);
        }
    }
}
