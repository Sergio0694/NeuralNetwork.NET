using System;
using System.Runtime.CompilerServices;
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
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static CostFunction GetCostFunction(CostFunctionType type)
        {
            switch (type)
            {
                case CostFunctionType.Quadratic: return CostFunctions.QuadraticCost;
                case CostFunctionType.CrossEntropy: return CostFunctions.CrossEntropyCost;
                case CostFunctionType.LogLikelyhood: return null; // TODO
                default:
                    throw new InvalidOperationException("Unsupported cost function");
            }
        }

        /// <summary>
        /// Gets the right cost function prime with the given type
        /// </summary>
        /// <param name="type">The cost function type</param>
        [Pure, NotNull]
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static CostFunctionPrime GetCostFunctionPrime(CostFunctionType type)
        {
            switch (type)
            {
                case CostFunctionType.Quadratic: return CostFunctions.QuadraticCostPrime;
                case CostFunctionType.CrossEntropy: return CostFunctions.CrossEntropyCostPrime;
                case CostFunctionType.LogLikelyhood: return null; // TODO
                default:
                    throw new InvalidOperationException("Unsupported cost function");
            }
        }
    }
}
