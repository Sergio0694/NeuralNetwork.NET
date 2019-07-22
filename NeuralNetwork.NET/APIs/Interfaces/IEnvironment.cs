using System;
using JetBrains.Annotations;

namespace NeuralNetworkNET.APIs.Interfaces
{
    /// <summary>
    /// An <see langword="interface"/> for an environment to use to train a network using reinforced learning
    /// </summary>
    public interface IEnvironment : IClonable<IEnvironment>, IEquatable<IEnvironment>, IDisposable
    {
        /// <summary>
        /// Gets the size of the current environment
        /// </summary>
        int Size { get; }

        /// <summary>
        /// Gets the number of available actions for the current environment
        /// </summary>
        int Actions { get; }

        /// <summary>
        /// Gets the reward for the current state of the environment
        /// </summary>
        int Reward { get; }

        /// <summary>
        /// Gets the timestep for the current environment
        /// </summary>
        int Timestep { get; }

        /// <summary>
        /// Executes an action with the specified index, and returns a new <see cref="IEnvironment"/> instance
        /// </summary>
        /// <param name="action">The index of the action to execute</param>
        /// <returns>The new, updated <see cref="IEnvironment"/> instance</returns>
        [Pure, NotNull]
        IEnvironment Execute(int action);

        /// <summary>
        /// Exports the current environment to a target <see cref="Span{T}"/>
        /// </summary>
        /// <param name="span">The target <see cref="Span{T}"/> instance to use to serialize the current environment</param>
        void Serialize(Span<float> span);
    }
}
