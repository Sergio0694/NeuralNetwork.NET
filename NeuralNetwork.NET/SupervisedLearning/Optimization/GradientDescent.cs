// ===================================
// Credits for the base of this code
// ===================================
//
// Accord Math Library
// The Accord.NET Framework
// http://accord-framework.net
//
// Copyright © César Souza, 2009-2017
// cesarsouza at gmail.com
//
//    This library is free software; you can redistribute it and/or
//    modify it under the terms of the GNU Lesser General Public
//    License as published by the Free Software Foundation; either
//    version 2.1 of the License, or (at your option) any later version.
//
//    This library is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
//    Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with this library; if not, write to the Free Software
//    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

using System;
using NeuralNetworkNET.SupervisedLearning.Optimization.Abstract;
using NeuralNetworkNET.SupervisedLearning.Optimization.Dependencies;

namespace NeuralNetworkNET.SupervisedLearning.Optimization
{
    /// <summary>
    ///   Gradient Descent (GD) for unconstrained optimization
    /// </summary>
    public class GradientDescent : GradientOptimizationMethodBase
    {
        // Gets the current convergence instance to monitor the progress
        private readonly RelativeConvergence Convergence = new RelativeConvergence();
        
        // Optimization parameter during the minimization
        private readonly int NumberOfUpdatesBeforeConvergenceCheck = 1;

        private double _LearningRate = 1e-3;

        /// <summary>
        ///   Gets or sets the learning rate. Default is 1e-3
        /// </summary>
        public double LearningRate
        {
            get => _LearningRate;
            set => _LearningRate = value <= 0 ? throw new ArgumentOutOfRangeException(nameof(LearningRate), "Learning rate should be higher than 0") : value;
        }

        /// <summary>
        ///   Gets or sets the maximum change in the average log-likelihood
        ///   after an iteration of the algorithm used to detect convergence.
        ///   Default is 1e-5
        /// </summary> 
        public double Tolerance
        {
            get => Convergence.Tolerance;
            set => Convergence.Tolerance = value;
        }

        /// <summary>
        ///   Gets or sets the maximum number of iterations
        ///   performed by the learning algorithm. Default is 0
        /// </summary>
        public int Iterations
        {
            get => Convergence.MaxIterations;
            set => Convergence.MaxIterations = value;
        }

        /// <summary>
        ///   Creates a new instance of the GD optimization algorithm
        /// </summary>
        public GradientDescent()
        {
            Iterations = 0;
            Tolerance = 1e-5;
        }

        /// <summary>
        ///   Implements the actual optimization algorithm. This
        ///   method should try to minimize the objective function
        /// </summary>
        protected override bool Optimize()
        {
            Convergence.Clear();
            int updates = 0;
            do
            {
                // Check the cancellation
                if (Token.IsCancellationRequested) break;

                // Perform the gradient descent
                double[] gradient = Gradient(Solution);
                for (int i = 0; i < Solution.Length; i++)
                    Solution[i] -= _LearningRate * gradient[i];

                // Check the progress
                updates++;
                if (updates >= NumberOfUpdatesBeforeConvergenceCheck)
                {
                    Convergence.NewValue = Function(Solution);
                    updates = 0;
                }
            }
            while (!Convergence.HasConverged);
            return true;
        }
    }
}
