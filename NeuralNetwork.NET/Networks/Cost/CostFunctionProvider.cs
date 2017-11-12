using System;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.Cost.Delegates;

namespace NeuralNetworkNET.Networks.Cost
{
    /// <summary>
    /// A static class that returns the right acost function for the given type
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
                case CostFunctionType.LogLikelyhood: return (null, null); // TODO
                default:
                    throw new InvalidOperationException("Unsupported cost function");
            }
        }
    }
}
