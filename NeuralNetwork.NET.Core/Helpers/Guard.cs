using System;
using System.Diagnostics;
using JetBrains.Annotations;

namespace NeuralNetwork.NET.Core.Helpers
{
    /// <summary>
    /// A <see langword="class"/> that exposes a set of APIs to check parameters in different methods across the library
    /// </summary>
    [DebuggerStepThrough]
    internal static class Guard
    {
        /// <summary>
        /// Ensures that the input condition is true
        /// </summary>
        /// <param name="value">The input condition</param>
        /// <param name="message">The message to display if the assertion fails</param>
        /// <exception cref="ArgumentException"><paramref name="value"/> is <see langword="false"/></exception>
        [AssertionMethod]
        public static void IsTrue([AssertionCondition(AssertionConditionType.IS_TRUE)] bool value, [NotNull] string message)
        {
            if (!value) throw new ArgumentException(message);
        }

        /// <summary>
        /// Ensures that the input condition is true
        /// </summary>
        /// <param name="value">The input condition</param>
        /// <param name="parameterName">The name of the parameter being checked</param>
        /// <param name="message">The message to display if the assertion fails</param>
        /// <exception cref="ArgumentException"><paramref name="value"/> is <see langword="false"/></exception>
        [AssertionMethod]
        public static void IsTrue([AssertionCondition(AssertionConditionType.IS_TRUE)] bool value, [NotNull] string parameterName, [NotNull] string message)
        {
            if (!value) throw new ArgumentException(message, parameterName);
        }

        /// <summary>
        /// Ensures that the input condition is false
        /// </summary>
        /// <param name="value">The input condition</param>
        /// <param name="message">The message to display if the assertion fails</param>
        /// <exception cref="ArgumentException"><paramref name="value"/> is <see langword="true"/></exception>
        [AssertionMethod]
        public static void IsFalse([AssertionCondition(AssertionConditionType.IS_FALSE)] bool value, [NotNull] string message)
        {
            if (value) throw new ArgumentException(message);
        }

        /// <summary>
        /// Ensures that the input condition is false
        /// </summary>
        /// <param name="value">The input condition</param>
        /// <param name="parameterName">The name of the parameter being checked</param>
        /// <param name="message">The message to display if the assertion fails</param>
        /// <exception cref="ArgumentException"><paramref name="value"/> is <see langword="true"/></exception>
        [AssertionMethod]
        public static void IsFalse([AssertionCondition(AssertionConditionType.IS_FALSE)] bool value, [NotNull] string parameterName, [NotNull] string message)
        {
            if (value) throw new ArgumentException(message, parameterName);
        }
    }
}
