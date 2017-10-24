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
using System.Threading;
using NeuralNetworkNET.Helpers;

namespace NeuralNetworkNET.SupervisedLearning.Optimization.Abstract
{
    /// <summary>
    ///   Base class for gradient-based optimization methods.
    /// </summary>
    public abstract class GradientOptimizationMethodBase : OptimizationMethodBase
    {

        /// <summary>
        ///   Gets or sets a function returning the gradient
        ///   vector of the function to be optimized for a
        ///   given value of its free parameters.
        /// </summary>
        /// 
        /// <value>The gradient function.</value>
        /// 
        public Func<double[], double[]> Gradient { get; set; }

        /// <summary>
        ///   Initializes a new instance of the <see cref="BaseGradientOptimizationMethod"/> class.
        /// </summary>
        /// 
        protected GradientOptimizationMethodBase()
            : base()
        {
        }

        /// <summary>
        ///   Initializes a new instance of the <see cref="BaseGradientOptimizationMethod"/> class.
        /// </summary>
        /// 
        /// <param name="numberOfVariables">The number of free parameters in the optimization problem.</param>
        /// 
        protected GradientOptimizationMethodBase(int numberOfVariables)
            : base(numberOfVariables)
        {
        }

        private static void CheckGradient(Func<double[], double[]> value, double[] probe)
        {
            double[] original = (double[])probe.Clone();
            double[] result = value(probe);

            if (result == probe)
                throw new InvalidOperationException(
                    "The gradient function should not return the parameter vector.");

            if (probe.Length != result.Length)
                throw new InvalidOperationException(
                    "The gradient vector should have the same length as the number of parameters.");

            for (int i = 0; i < probe.Length; i++)
                if (!probe[i].EqualsWithDelta(original[i]))
                    throw new InvalidOperationException("The gradient function shouldn't modify the parameter vector.");
        }

        /// <summary>
        ///   Finds the minimum value of a function. The solution vector
        ///   will be made available at the <see cref="IOptimizationMethod{TInput, TOutput}.Solution"/> property.
        /// </summary>
        /// 
        /// <returns>Returns <c>true</c> if the method converged to a <see cref="IOptimizationMethod{TInput, TOutput}.Solution"/>.
        ///   In this case, the found value will also be available at the <see cref="IOptimizationMethod{TInput, TOutput}.Value"/>
        ///   property.</returns>
        ///  
        public override bool Minimize()
        {
            if (Gradient == null)
                throw new ArgumentNullException("The gradient function can't be null");

            CheckGradient(Gradient, Solution);

            return base.Minimize();
        }

    }
}
