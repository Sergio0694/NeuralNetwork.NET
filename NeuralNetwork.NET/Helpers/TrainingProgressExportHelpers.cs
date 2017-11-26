using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using JetBrains.Annotations;
using NeuralNetworkNET.Networks.PublicAPIs;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A class with some helper methods to quickly convert a training report to a 2D chart
    /// </summary>
    public static class TrainingProgressExportHelpers
    {
        /// <summary>
        /// Returns a Python script to plot a 2D chart from the given progress reports
        /// </summary>
        /// <param name="results">The input results to plot</param>
        /// <param name="type">The type of progress report to plot</param>
        public static String AsPythonMatplotlibChart([NotNull] this IReadOnlyList<DatasetEvaluationResult> results, TrainingReportType type)
        {
            String
                values = results
                    .Select((r, i) => (Result: r, Index: i))
                    .Aggregate(String.Empty, (b, v) => $"{b}{v.Result[type].ToString(CultureInfo.InvariantCulture)}{(v.Index == results.Count - 1 ? String.Empty : ",\n    ")}"),
                ylabel = type == TrainingReportType.Accuracy ? "Accuracy" : "Cost",
                template = File.ReadAllText(Path.Combine(Path.GetDirectoryName(Path.GetFullPath(Assembly.GetExecutingAssembly().Location)), "Assets", "PltTemplate.py"));
            return template.Replace("$VALUES$", values).Replace("$YLABEL$", ylabel);
        }
    }

    /// <summary>
    /// Indicates a tye of training progress report
    /// </summary>
    public enum TrainingReportType
    {
        Accuracy,
        Cost
    }
}
