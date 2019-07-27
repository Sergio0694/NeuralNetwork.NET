using System;
using System.Collections.Generic;
using JetBrains.Annotations;

namespace NeuralNetworkNET.ReinforcedLearning
{
    /// <summary>
    /// A static class that produces different decaying rates to be used during training
    /// </summary>
    public static class DecayingRates
    {
        /// <summary>
        /// Creates a new exponential decaying enumerator
        /// </summary>
        /// <param name="decay">The decay factor to use</param>
        [Pure, NotNull]
        public static IEnumerator<float> Exponential(float decay)
        {
            var i = 0;
            while (true)
            {
                yield return (float)Math.Exp(-i / decay);
                i++;
            }
        }

        /// <summary>
        /// Creates a new linear decaying enumerator
        /// </summary>
        /// <param name="factor">The decaying factor to use</param>
        [Pure, NotNull]
        public static IEnumerator<float> Linear(float factor)
        {
            var value = 1f;
            while (true)
            {
                yield return value;
                value *= factor;
            }
        }
    }
}
