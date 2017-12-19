using JetBrains.Annotations;
using System;

namespace NeuralNetworkNET.DependencyInjections
{
    /// <summary>
    /// A static class with some static delagates used to inject additional functionalities from external, linked libraries
    /// </summary>
    internal static class LibraryRuntimeHelper
    {
        /// <summary>
        /// An <see cref="Action"/> that is executed right before the training starts on a network
        /// </summary>
        [CanBeNull]
        public static Action SynchronizeContext;
    }
}
