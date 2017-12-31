using JetBrains.Annotations;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.Services
{
    /// <summary>
    /// A static class with events that signal different status changes for the library
    /// </summary>
    internal static class SharedEventsService
    {
        /// <summary>
        /// An <see cref="System.Action"/> that is executed right before the training starts on a network
        /// </summary>
        [NotNull]
        public static readonly SharedEvent TrainingStarting = new SharedEvent();
    }
}
