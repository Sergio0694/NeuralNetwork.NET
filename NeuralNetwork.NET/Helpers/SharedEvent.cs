using System;
using System.Linq;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A shared event with an <see cref="Action"/> <see langword="delegate"/> as its handler
    /// </summary>
    internal sealed class SharedEvent
    {
        // The backing delegate
        [CanBeNull]
        private Action _InvocationList;

        /// <summary>
        /// Adds the input <see cref="Action"/> to the list of handlers
        /// </summary>
        /// <param name="action">The <see cref="Action"/> to add</param>
        public void Add(Action action)
        {
            if (_InvocationList?.GetInvocationList().Contains(action) == true) return;
            _InvocationList += action;
        }

        /// <summary>
        /// Raises the shared event
        /// </summary>
        public void Raise() => _InvocationList?.Invoke();
    }
}
