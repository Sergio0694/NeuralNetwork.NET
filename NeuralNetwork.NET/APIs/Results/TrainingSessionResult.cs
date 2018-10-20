using System;
using System.Collections.Generic;
using JetBrains.Annotations;
using NeuralNetworkNET.APIs.Enums;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace NeuralNetworkNET.APIs.Results
{
    /// <summary>
    /// A class that contains all the information on a completed training session
    /// </summary>
    [JsonObject(MemberSerialization.OptOut)]
    public sealed class TrainingSessionResult
    {
        /// <summary>
        /// Gets the result for the training session
        /// </summary>
        public TrainingStopReason StopReason { get; }

        /// <summary>
        /// Gets the number of completed training epochs
        /// </summary>
        public int CompletedEpochs { get; }

        /// <summary>
        /// Gets the approximate training time for the current session
        /// </summary>
        public TimeSpan TrainingTime { get; }

        /// <summary>
        /// Gets the evaluation reports for the validation dataset, if provided
        /// </summary>
        [NotNull]
        public IReadOnlyList<DatasetEvaluationResult> ValidationReports { get; }

        /// <summary>
        /// Gets the evaluation reports for the test set, if provided
        /// </summary>
        [NotNull]
        public IReadOnlyList<DatasetEvaluationResult> TestReports { get; }

        /// <summary>
        /// Serializes the current instance as a JSON string with all the current training info
        /// </summary>
        [Pure, NotNull]
        public string SerializeAsJson() => JsonConvert.SerializeObject(this, Formatting.Indented, new StringEnumConverter());

        // Internal constructor
        internal TrainingSessionResult(
            TrainingStopReason stopReason, int epochs, TimeSpan time,
            [NotNull] IReadOnlyList<DatasetEvaluationResult> validationReports,
            [NotNull] IReadOnlyList<DatasetEvaluationResult> testReports)
        {
            StopReason = stopReason;
            CompletedEpochs = epochs;
            TrainingTime = time;
            ValidationReports = validationReports;
            TestReports = testReports;
        }
    }
}
