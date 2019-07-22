using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Interfaces;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.ReinforcedLearning.Environments
{
    /// <summary>
    /// An environment for the tic-tac-toe game
    /// </summary>
    public sealed class TicTacToeEnvironment : IEnvironment
    {
        /// <inheritdoc/>
        public int Size { get; } = 9;

        /// <inheritdoc/>
        public int Actions { get; } = 9;

        /// <inheritdoc/>
        public int Reward => Timestep;

        /// <inheritdoc/>
        public int Timestep { get; }

        /// <inheritdoc/>
        public bool CanExecute
        {
            get
            {
                ref float r = ref State[0];
                for (int i = 0; i < 9; i++)
                    if (0f.EqualsWithDelta(Unsafe.Add(ref r, i)))
                        return true;
                return false;
            }
        }

        /// <summary>
        /// A singleton <see cref="ArrayPool{T}"/> instance to quickly create new instances of the environment
        /// </summary>
        [NotNull]
        private static readonly ArrayPool<float> Allocator = ArrayPool<float>.Create(9, 1000);

        /// <summary>
        /// The array representing the environment current state
        /// </summary>
        [NotNull]
        private readonly float[] State = Allocator.Rent(9);

        /// <summary>
        /// Creates a new, empty <see cref="TicTacToeEnvironment"/> instance
        /// </summary>
        public TicTacToeEnvironment() { }

        private TicTacToeEnvironment(int timestep) { Timestep = timestep; }

        /// <inheritdoc/>
        public IEnvironment Execute(int action)
        {
            bool valid = State[action].EqualsWithDelta(0);
            var instance = new TicTacToeEnvironment(Timestep + 1);
            if (valid) State[action] = Timestep % 2 == 0 ? 1 : -1;
            return instance;
        }

        /// <inheritdoc/>
        public void Serialize(Span<float> span) => State.AsSpan(0, Size).CopyTo(span);

        /// <inheritdoc/>
        public IEnvironment Clone()
        {
            var instance = new TicTacToeEnvironment(Timestep);
            State.AsSpan(0, Size).CopyTo(instance.State.AsSpan(0, Size));
            return instance;
        }

        /// <inheritdoc/>
        public bool Equals(IEnvironment other)
        {
            return other is TicTacToeEnvironment instance &&
                   State.AsSpan(0, Size).ContentEquals(instance.State.AsSpan(0, Size));
        }

        /// <inheritdoc/>
        ~TicTacToeEnvironment() => Allocator.Return(State);

        /// <inheritdoc/>
        public void Dispose()
        {
            Allocator.Return(State);
            GC.SuppressFinalize(this);
        }
    }
}
