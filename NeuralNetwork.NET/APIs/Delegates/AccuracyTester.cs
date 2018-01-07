using System;

namespace NeuralNetworkNET.APIs.Delegates
{
    /// <summary>
    /// A <see langword="delegate"/> that wraps a function used to test whether or not the outputs for an evaluation sample match the expected results
    /// </summary>
    /// <param name="yHat">The sample outputs produced by the current network</param>
    /// <param name="y">The expected outputs for the sample</param>
    public delegate bool AccuracyTester(Span<float> yHat, Span<float> y);
}
