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
    ///   Base class for optimization methods.
    /// </summary>
    public abstract class OptimizationMethodBase
    {
        private int numberOfVariables;
        private double[] x;
        private double value;

        [NonSerialized]
        private CancellationToken token = new CancellationToken();

        /// <summary>
        ///   Gets or sets a cancellation token that can be used to
        ///   stop the learning algorithm while it is running.
        /// </summary>
        /// 
        public CancellationToken Token
        {
            get { return token; }
            set { token = value; }
        }

        /// <summary>
        ///   Gets or sets the function to be optimized.
        /// </summary>
        /// 
        /// <value>The function to be optimized.</value>
        /// 
        public Func<double[], double> Function { get; set; }

        /// <summary>
        ///   Gets the number of variables (free parameters)
        ///   in the optimization problem.
        /// </summary>
        /// 
        /// <value>The number of parameters.</value>
        /// 
        public virtual int NumberOfVariables
        {
            get { return numberOfVariables; }
            set
            {
                this.numberOfVariables = value;
                OnNumberOfVariablesChanged(value);
            }
        }

        /// <summary>
        ///   Gets the current solution found, the values of 
        ///   the parameters which optimizes the function.
        /// </summary>
        /// 
        public double[] Solution
        {
            get { return x; }
            set
            {
                if (value == null)
                    throw new ArgumentNullException("value");

                if (value.Length != NumberOfVariables)
                    throw new ArgumentException("value");

                x = value;
            }
        }

        /// <summary>
        ///   Gets the output of the function at the current <see cref="Solution"/>.
        /// </summary>
        /// 
        public double Value
        {
            get { return value; }
            protected set { this.value = value; }
        }

        /// <summary>
        ///   Initializes a new instance of the <see cref="BaseOptimizationMethod"/> class.
        /// </summary>
        /// 
        protected OptimizationMethodBase()
        {
        }

        /// <summary>
        ///   Initializes a new instance of the <see cref="BaseOptimizationMethod"/> class.
        /// </summary>
        /// 
        /// <param name="numberOfVariables">The number of free parameters in the optimization problem.</param>
        /// 
        protected OptimizationMethodBase(int numberOfVariables)
        {
            if (numberOfVariables <= 0)
                throw new ArgumentOutOfRangeException("numberOfVariables");

            this.NumberOfVariables = numberOfVariables;
        }

        /// <summary>
        ///   Initializes a new instance of the <see cref="BaseOptimizationMethod"/> class.
        /// </summary>
        /// 
        /// <param name="numberOfVariables">The number of free parameters in the optimization problem.</param>
        /// <param name="function">The objective function whose optimum values should be found.</param>
        /// 
        protected OptimizationMethodBase(int numberOfVariables, Func<double[], double> function)
        {
            if (function == null)
                throw new ArgumentNullException("function");

            this.NumberOfVariables = numberOfVariables;
            this.Function = function;
        }

        /// <summary>
        /// Called when the <see cref="NumberOfVariables"/> property has changed.
        /// </summary>
        /// 
        protected virtual void OnNumberOfVariablesChanged(int numberOfVariables)
        {
            Random random = new Random();
            if (this.Solution == null || this.Solution.Length != numberOfVariables)
            {
                this.Solution = new double[numberOfVariables];
                for (int i = 0; i < Solution.Length; i++)
                    Solution[i] = random.NextGaussian();
            }
        }

        /// <summary>
        ///   Finds the maximum value of a function. The solution vector
        ///   will be made available at the <see cref="Solution"/> property.
        /// </summary>
        /// 
        /// <param name="values">The initial solution vector to start the search.</param>
        /// 
        /// <returns>Returns <c>true</c> if the method converged to a <see cref="Solution"/>.
        ///   In this case, the found value will also be available at the <see cref="Value"/>
        ///   property.</returns>
        ///  
        public bool Maximize(double[] values)
        {
            Solution = values;
            return Maximize();
        }


        /// <summary>
        ///   Finds the minimum value of a function. The solution vector
        ///   will be made available at the <see cref="Solution"/> property.
        /// </summary>
        /// 
        /// <param name="values">The initial solution vector to start the search.</param>
        /// 
        /// <returns>Returns <c>true</c> if the method converged to a <see cref="Solution"/>.
        ///   In this case, the found value will also be available at the <see cref="Value"/>
        ///   property.</returns>
        ///  
        public bool Minimize(double[] values)
        {
            Solution = values;
            return Minimize();
        }

        /// <summary>
        ///   Finds the maximum value of a function. The solution vector
        ///   will be made available at the <see cref="Solution"/> property.
        /// </summary>
        /// 
        /// <returns>Returns <c>true</c> if the method converged to a <see cref="Solution"/>.
        ///   In this case, the found value will also be available at the <see cref="Value"/>
        ///   property.</returns>
        ///  
        public virtual bool Maximize()
        {
            if (Function == null)
                throw new InvalidOperationException("function");

            var f = Function;

            Function = (x) => -f(x);

            bool success = Optimize();

            Function = f;

            value = Function(Solution);

            return success;
        }


        /// <summary>
        ///   Finds the minimum value of a function. The solution vector
        ///   will be made available at the <see cref="Solution"/> property.
        /// </summary>
        /// 
        /// <returns>Returns <c>true</c> if the method converged to a <see cref="Solution"/>.
        ///   In this case, the found value will also be available at the <see cref="Value"/>
        ///   property.</returns>
        ///  
        public virtual bool Minimize()
        {
            if (Function == null)
                throw new InvalidOperationException("function");

            bool success = Optimize();

            value = Function(Solution);

            return success;
        }


        /// <summary>
        ///   Implements the actual optimization algorithm. This
        ///   method should try to minimize the objective function.
        /// </summary>
        /// 
        protected abstract bool Optimize();


        /// <summary>
        ///   Creates an exception with a given inner optimization algorithm code (for debugging purposes).
        /// </summary>
        /// 
        protected static ArgumentOutOfRangeException ArgumentException(string paramName, string message, string code)
        {
            var e = new ArgumentOutOfRangeException(paramName, message);
            e.Data["Code"] = code;
            return e;
        }

        /// <summary>
        ///   Creates an exception with a given inner optimization algorithm code (for debugging purposes).
        /// </summary>
        /// 
        protected static InvalidOperationException OperationException(string message, string code)
        {
            var e = new InvalidOperationException(message);
            e.Data["Code"] = code;
            return e;
        }
    }
}