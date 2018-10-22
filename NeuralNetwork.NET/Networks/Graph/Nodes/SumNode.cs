using System;
using System.Collections.Generic;
using Alea;
using Alea.cuDNN;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.APIs.Structs;
using NeuralNetworkNET.cpuDNN;
using NeuralNetworkNET.cuDNN;
using NeuralNetworkNET.Extensions;
using NeuralNetworkNET.Networks.Activations;
using NeuralNetworkNET.Networks.Activations.Delegates;
using NeuralNetworkNET.Networks.Graph.Nodes.Abstract;

namespace NeuralNetworkNET.Networks.Graph.Nodes
{
    /// <summary>
    /// A class representing a sum node in a computation graph
    /// </summary>
    internal abstract class SumNode : MergeNodeBase
    {
        #region Initialization

        /// <summary>
        /// Gets the activation type used in the current node
        /// </summary>
        public ActivationType ActivationType { get; }

        /// <summary>
        /// Gets the list of activation and activation prime functions used in the sum node
        /// </summary>
        private readonly (ActivationFunction Activation, ActivationFunction ActivationPrime) ActivationFunctions;

        /// <summary>
        /// Gets the execution mode set for this <see cref="SumNode"/> instance
        /// </summary>
        public ExecutionModePreference ExecutionMode { get; }

        protected SumNode(ExecutionModePreference mode, ActivationType activation, [NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) : base(ComputationGraphNodeType.Sum, parents)
        {
            ExecutionMode = mode;
            ActivationType = activation;
            ActivationFunctions = ActivationFunctionProvider.GetActivations(activation);
        }

        /// <summary>
        /// Creates a new <see cref="SumNode"/> with the given parameters
        /// </summary>
        /// <param name="activation">The sum node activation function</param>
        /// <param name="mode">The desired execution mode</param>
        /// <param name="parents">The parent nodes for the new sum mode to create</param>
        [Pure, NotNull]
        public static SumNode New(ActivationType activation, ExecutionModePreference mode, [NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents)
        {
            if (mode == ExecutionModePreference.Cpu) return new CpuSumNode(activation, parents);
            return new CudaSumNode(activation, parents);
        }

        #endregion

        /// <summary>
        /// Forwards the inputs through the graph node and returns the resulting activity (Z) and activation (A)
        /// </summary>
        /// <param name="inputs">The inputs to process</param>
        /// <param name="z">The output activity on the current node</param>
        /// <param name="a">The output activation on the current node</param>
        public abstract void Forward(Span<Tensor> inputs, out Tensor z, out Tensor a);

        /// <summary>
        /// Backpropagates the error to compute the delta for the inputs of the graph node
        /// </summary>
        /// <param name="y">The output <see cref="Tensor"/> computed in the forward pass</param>
        /// <param name="dy">The output error delta to backpropagate</param>
        /// <param name="dx">The resulting backpropagated error</param>
        public abstract void Backpropagate(in Tensor y, in Tensor dy, in Tensor dx);

        #region Implementation

        /// <summary>
        /// A CPU-powered sum node
        /// </summary>
        private sealed class CpuSumNode : SumNode
        {
            public CpuSumNode(ActivationType activation, [NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) 
                : base(ExecutionModePreference.Cpu, activation, parents) { }

            /// <inheritdoc/>
            public override void Forward(Span<Tensor> inputs, out Tensor z, out Tensor a)
            {
                Tensor.New(inputs[0].Entities, inputs[0].Length, out z);
                CpuBlas.Sum(inputs, z);
                Tensor.Like(z, out a);
                CpuDnn.ActivationForward(z, ActivationFunctions.Activation, a);
            }

            /// <inheritdoc/>
            public override void Backpropagate(in Tensor y, in Tensor dy, in Tensor dx)
                => CpuDnn.ActivationBackward(y, dy, ActivationFunctions.ActivationPrime, dx);
        }

        /// <summary>
        /// A CUDA-powered sum node
        /// </summary>
        private sealed class CudaSumNode : SumNode
        {
            // The NCHW tensor info for the node inputs and output
            [NotNull]
            private readonly TensorDescriptor Descriptor = new TensorDescriptor();

            /// <summary>
            /// Gets the <see cref="Dnn"/> instance for the current node
            /// </summary>
            [NotNull]
            private readonly Dnn DnnInstance = CuDnnService.Instance;

            public CudaSumNode(ActivationType activation, [NotNull] [ItemNotNull] IReadOnlyList<IComputationGraphNode> parents) 
                : base(ExecutionModePreference.Cuda, activation, parents) { }

            /// <inheritdoc/>
            public override unsafe void Forward(Span<Tensor> inputs, out Tensor z, out Tensor a)
            {
                Descriptor.Set4D(DataType.FLOAT, TensorFormat.CUDNN_TENSOR_NCHW, inputs[0].Entities, inputs[0].Length, 1, 1);
                fixed (Tensor* p = inputs)
                {
                    Tensor.New(p->Entities, p->Length, out z);
                    using (DeviceMemory<float> y_gpu = DnnInstance.Gpu.AllocateDevice(*p))
                    {
                        // Sum the inputs
                        for (int i = 1; i < inputs.Length; i++)
                            using (DeviceMemory<float> x_gpu = DnnInstance.Gpu.AllocateDevice(p[i]))
                                DnnInstance.AddTensor(1, Descriptor, x_gpu.Ptr, 1, Descriptor, y_gpu.Ptr);
                        y_gpu.CopyToHost(p[0].Entities, p[0].Length, out z);

                        // Activation
                        DnnInstance.ActivationForward(p[0].Entities, p[0].Length, y_gpu.Ptr, y_gpu.Ptr, ActivationFunctions.Activation);
                        y_gpu.CopyToHost(z.Entities, z.Length, out a);
                    }
                }
            }

            /// <inheritdoc/>
            public override void Backpropagate(in Tensor y, in Tensor dy, in Tensor dx)
            {
                using (DeviceMemory<float>
                    y_gpu = DnnInstance.Gpu.AllocateDevice(y),
                    dy_gpu = DnnInstance.Gpu.AllocateDevice(dy))
                {
                    DnnInstance.ActivationBackward(y.Entities, y.Length, y_gpu.Ptr, dy_gpu.Ptr, ActivationFunctions.ActivationPrime, dy_gpu.Ptr);
                    dy_gpu.CopyTo(dx);
                }
            }
        }

        #endregion

        /// <inheritdoc/>
        public override void Serialize(System.IO.Stream stream)
        {
            base.Serialize(stream);
            stream.Write(ActivationType);
        }
    }
}
