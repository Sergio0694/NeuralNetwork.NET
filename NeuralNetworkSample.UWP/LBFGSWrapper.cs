using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using NeuralNetworkNET;

namespace NeuralNetworkSample.UWP
{
    /// <summary>
    /// A simple wrapper for the LBFGS method in Accord.Math
    /// </summary>
    public sealed class LBFGSWrapper
    {
        // The LBFGS instance
        private readonly BoundedBroydenFletcherGoldfarbShanno LBFGS;

        // Simple constructor
        public LBFGSWrapper(int variables, Func<double[], double> cost, Func<double[], double[]> gradient)
        {
            LBFGS = new BoundedBroydenFletcherGoldfarbShanno(variables, cost, gradient);
            LBFGS.Progress += (s, e) => ProgressRelay?.Invoke(this,
                new IAccordNETGradientOptimizationMethodProgressRelayEventArgs(e.Solution, e.Iteration, e.Value));
        }

        public void Minimize() => LBFGS.Minimize();

        public void Minimize(double[] solution) => LBFGS.Minimize(solution);

        public CancellationToken Token
        {
            get => LBFGS.Token;
            set => LBFGS.Token = value;
        }

        public double[] Solution => LBFGS.Solution;

        public event EventHandler<IAccordNETGradientOptimizationMethodProgressRelayEventArgs> ProgressRelay;
    }
}
