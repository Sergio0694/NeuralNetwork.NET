using System;
using System.Threading;
using JetBrains.Annotations;

namespace NeuralNetwork.NET
{
    /// <summary>
    /// A delegate that takes the number of variables in a function to optimize and returns an instance of the LBFGS method
    /// </summary>
    /// <param name="numberOfVariables">The number of free variables to optimize</param>
    /// <param name="costFunction">The function to minimize</param>
    /// <param name="gradientFunction">A function that returns the gradient for the function to minimize</param>
    public delegate IAccordNETGradientOptimizationMethod LBFGSFactory(
        int numberOfVariables,
        [NotNull] Func<double[], double> costFunction,
        [NotNull] Func<double[], double[]> gradientFunction);

    /// <summary>
    /// COMPATIBILITY LAYER: interface to wrap an (Accord.NET).Math.Optimization.BoundedBroydenFletcherGoldfarbShanno instance.
    /// This interface will be removed once .NET Standard 2.0 is released and the library will be able to reference the Accord.Math
    /// library internally (this is a temporary workaround to mantain platform/framework independence)
    /// </summary>
    public interface IAccordNETGradientOptimizationMethod
    {
        /// <summary>
        /// Forwards the call to the Minimize() method of the Accord.Math LBFGS instance
        /// </summary>
        void Minimize();

        /// <summary>
        /// Forwards the call to the Minimize(double[] solution) method of the Accord.Math LBFGS instance
        /// </summary>
        /// <param name="solution">The current solution to start with</param>
        void Minimize(double[] solution);

        /// <summary>
        /// The cancellation token to pass to the LBFGS instance in use
        /// </summary>
        CancellationToken Token { get; set; }

        /// <summary>
        /// An event that relays the Progress event of the LBFGS instance in use
        /// </summary>
        event EventHandler<IAccordNETGradientOptimizationMethodProgressRelayEventArgs> ProgressRelay;
    }

    /// <summary>
    /// The arguments for the ProgressRelay event
    /// </summary>
    public class IAccordNETGradientOptimizationMethodProgressRelayEventArgs : EventArgs
    {
        /// <summary>
        /// Creates a new instance of the arguments with the given parameters (to get from 
        /// the OptimizationProgressEventArgs arguments in the LBFGS Progress event)
        /// </summary>
        /// <param name="solution">The current solution found</param>
        /// <param name="iteration">The current iteration</param>
        /// <param name="value">The current optimized value</param>
        public IAccordNETGradientOptimizationMethodProgressRelayEventArgs(
            [NotNull] double[] solution, int iteration, double value)
        {
            Solution = solution;
            Iteration = iteration;
            Value = value;
        }

        /// <summary>
        /// Gets the current solution for the function to minimize
        /// </summary>
        [NotNull]
        public double[] Solution { get; }

        /// <summary>
        /// Gets the current optimization iteration
        /// </summary>
        public int Iteration { get; }

        /// <summary>
        /// Gets the current value of the minimized function
        /// </summary>
        public double Value { get; }
    }
}
