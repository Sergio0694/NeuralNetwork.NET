using System;
using JetBrains.Annotations;
using NeuralNetworkDotNet.APIs.Enums;
using NeuralNetworkDotNet.Network.Cost.Delegates;

namespace NeuralNetworkDotNet.Network.Cost
{
    /// <summary>
    /// A <see langword="class"/> that returns the right cost function for the given type
    /// </summary>
    internal static class CostFunctionProvider
    {
        /// <summary>
        /// Gets the right cost function with the given type
        /// </summary>
        /// <param name="type">The cost function type</param>
        [Pure]
        public static (CostFunction, CostFunctionPrime) GetCostFunctions(CostFunctionType type)
        {
            switch (type)
            {
                case CostFunctionType.Quadratic: return (CostFunctions.QuadraticCost, CostFunctions.QuadraticCostPrime);
                case CostFunctionType.CrossEntropy: return (CostFunctions.CrossEntropyCost, CostFunctions.CrossEntropyCostPrime);
                case CostFunctionType.LogLikelyhood: return (CostFunctions.LogLikelyhoodCost, CostFunctions.CrossEntropyCostPrime);
                default: throw new InvalidOperationException($"Unsupported cost function: {type}");
            }
        }
    }
}
