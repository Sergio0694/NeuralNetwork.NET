using System.IO;
using JetBrains.Annotations;

namespace NeuralNetworkDotNet.APIs.Interfaces
{
    /// <summary>
    /// An interface for an object that supports serialization to a target <see cref="Stream"/>
    /// </summary>
    public interface ISerializable
    {
        /// <summary>
        /// Serializes the current instance into a target <see cref="Stream"/> instance
        /// </summary>
        /// <param name="stream">The target <see cref="Stream"/> to write to</param>
        void Serialize([NotNull] Stream stream);
    }
}
