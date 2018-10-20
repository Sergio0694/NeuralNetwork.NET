using System;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;

namespace NeuralNetworkNET.Extensions
{
    /// <summary>
    /// A simple class with some extension methods for the <see cref="HttpClient"/> class
    /// </summary>
    public static class HttpClientExtensions
    {
        /// <summary>
        /// Downloads a <see cref="Stream"/> from the given URL, and reports the download progress using the input callback
        /// </summary>
        /// <param name="client">The <see cref="HttpClient"/> instance to use to download the data</param>
        /// <param name="url">The URL to download</param>
        /// <param name="callback">The optional progress calback</param>
        /// <param name="token">The optional token for the download operation</param>
        [MustUseReturnValue, NotNull, ItemCanBeNull]
        public static async Task<Stream> GetAsync([NotNull] this HttpClient client, string url, [CanBeNull] IProgress<HttpProgress> callback, CancellationToken token = default)
        {
            using (HttpResponseMessage response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, token))
            {
                if (!response.IsSuccessStatusCode || token.IsCancellationRequested) return null;
                using (Stream source = await response.Content.ReadAsStreamAsync())
                {
                    // Read and store the data
                    Stream result = new MemoryStream();
                    long
                        totalRead = 0L,
                        totalReads = 0L,
                        length = response.Content.Headers.ContentLength ?? 0;
                    byte[] buffer = new byte[8192];
                    bool isMoreToRead = true;
                    do
                    {
                        int read = await source.ReadAsync(buffer, 0, buffer.Length, token);
                        if (read == 0) isMoreToRead = false;
                        else
                        {
                            await result.WriteAsync(buffer, 0, read, token);
                            totalRead += read;
                            if (totalReads++ % 2000 == 0) 
                                callback?.Report(new HttpProgress(totalRead, length > 0 ? (int)(totalRead * 100 / length) : 0));
                        }
                    }
                    while (isMoreToRead && !token.IsCancellationRequested);

                    // Return the result
                    if (token.IsCancellationRequested)
                    {
                        result.Dispose();
                        return null;
                    }
                    result.Seek(0, SeekOrigin.Begin); // Move the content stream back to the start
                    return result;
                }
            }
        }
    }

    /// <summary>
    /// A <see langword="struct"/> that contains info on a pending download
    /// </summary>
    public readonly struct HttpProgress
    {
        /// <summary>
        /// Gets the total number of downloaded bytes
        /// </summary>
        public long DownloadedBytes { get; }

        /// <summary>
        /// Gets the current download percentage
        /// </summary>
        public int Percentage { get; }

        internal HttpProgress(long bytes, int percentage)
        {
            DownloadedBytes = bytes;
            Percentage = percentage;
        }
    }
}
