using System;
using System.IO;
using System.Net.Http;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using JetBrains.Annotations;
using NeuralNetworkNET.Extensions;

namespace NeuralNetworkNET.Helpers
{
    /// <summary>
    /// A static class that handles in-app resources downloaded from the web
    /// </summary>
    internal static class DatasetsDownloader
    {
        #region Fields and properties

        // The default file extension for local resource files
        private const String FileExtension = ".data";

        /// <summary>
        /// Gets the default datasets path to use to store and load fdata files
        /// </summary>
        [NotNull]
        private static String DatasetsPath
        {
            get
            {
                String
                    code = Assembly.GetExecutingAssembly().Location,
                    dll = Path.GetFullPath(code),
                    root = Path.GetDirectoryName(dll),
                    path = Path.Combine(root, "Datasets");
                return path;
            }
        }

        // Local lazy instance of the singleton HttpClient in use
        [NotNull]
        private static readonly Lazy<HttpClient> _Client = new Lazy<HttpClient>(() => new HttpClient());

        /// <summary>
        /// Gets the singleton <see cref="HttpClient"/> to use, since it is reentrant and thread-safe, see <a href="https://docs.microsoft.com/it-it/dotnet/api/system.net.http.httpclient">docs.microsoft.com/it-it/dotnet/api/system.net.http.httpclient</a>
        /// </summary>
        [NotNull]
        private static HttpClient Client
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get
            {
                lock (_Client)
                    return _Client.Value;
            }
        }

        #endregion

        /// <summary>
        /// Gets a <see cref="Stream"/> with the contents of the input URL
        /// </summary>
        /// <param name="url">The target URL to use to download the resources</param>
        /// <param name="token">A cancellation token for the operation</param>
        public static async Task<Stream> GetAsync([NotNull] String url, CancellationToken token)
        {
            // Check if the target resource already exists
            byte[]
                bytes = Encoding.UTF8.GetBytes(url),
                hash = HashAlgorithm.Create(HashAlgorithmName.MD5.Name).TransformFinalBlock(bytes, 0, bytes.Length);
            String
                filename = $"{hash.ToBase16()}{FileExtension}",
                path = Path.Combine(DatasetsPath, filename);
            if (File.Exists(path)) return File.OpenRead(path);

            try
            {
                // Download from the input URL
                HttpResponseMessage result = await Client.GetAsync(url, token);
                if (!result.IsSuccessStatusCode || token.IsCancellationRequested) return null;
                byte[] data = await result.Content.ReadAsByteArrayAsync();

                // Write the HTTP content
                FileStream stream = File.OpenWrite(path);
                await stream.WriteAsync(data, 0, data.Length, default); // Ensure the whole content is written to disk
                stream.Seek(0, SeekOrigin.Begin);
                return stream;
            }
            catch
            {
                // Connection error or operation canceled by the user
                return null;
            }
        }
    }
}
