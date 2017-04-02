using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.Win32;

namespace DigitsRecognitionSample
{
    /// <summary>
    /// Logica di interazione per MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            this.MouseMove += (s, e) =>
            {
                if (e.LeftButton == MouseButtonState.Pressed) this.DragMove();
            };
        }

        private void OnTopToggle_OnOnToggled(object sender, EventArgs e) => Topmost = !Topmost;

        private void MinimizeButton_Clicked(object sender, RoutedEventArgs e) => WindowState = WindowState.Minimized;

        private void CloseButton_Clicked(object sender, RoutedEventArgs e) => Application.Current.Shutdown();

        #region Navigation

        /// <summary>
        /// Manages the navigation by making sure no double instances are created
        /// </summary>
        /// <typeparam name="T">The page to open</typeparam>
        private void NavigatePage<T>() where T : Window, new()
        {
            Window window = Application.Current.Windows.OfType<Window>().SingleOrDefault(x => x.GetType() == typeof(T));
            if (window == null)
            {
                new T { Left = this.Left + 25, Top = this.Top + 25 }.Show();
            }
            else
            {
                window.Focus();
            }
        }

        #endregion

        private void Button_Click(object sender, RoutedEventArgs e)
        {
            OpenFileDialog picker = new OpenFileDialog
            {
                DefaultExt = ".png",
                Filter = "PNG image (.png)|*.png"
            };
            bool? result = picker.ShowDialog();
            if (result.HasValue && result.Value)
            {
                Bitmap 
                    image = new Bitmap(picker.FileName),
                    grayscale = ToGrayscale(image);
                SaveFileDialog saver = new SaveFileDialog
                {
                    DefaultExt = ".png",
                    Filter = "PNG image (.png)|*.png"
                };
                result = saver.ShowDialog();
                if (result.HasValue && result.Value)
                {
                    grayscale.Save(saver.FileName);
                }
            }
        }

        public static Bitmap ToGrayscale(Bitmap original)
        {
            // Create a blank bitmap the same size as original
            Bitmap newBitmap = new Bitmap(original.Width, original.Height);

            // Get a graphics object from the new image
            using (Graphics g = Graphics.FromImage(newBitmap))
            {
                // Create the grayscale ColorMatrix
                ColorMatrix colorMatrix = new ColorMatrix(
                new float[][]
                {
                    new float[] { 0.3f, 0.3f, 0.3f, 0, 0 },
                    new float[] { 0.59f, 0.59f, 0.59f, 0, 0 },
                    new float[] { 0.11f, 0.11f, 0.11f, 0, 0 },
                    new float[] { 0, 0, 0, 1, 0 },
                    new float[] { 0, 0, 0, 0, 1 }
                });

                // Create the image attributes and set the color matrix
                ImageAttributes attributes = new ImageAttributes();
                attributes.SetColorMatrix(colorMatrix);

                // Draw the original image on the new image using the grayscale color matrix
                g.DrawImage(original, new System.Drawing.Rectangle(0, 0, original.Width, original.Height),
                    0, 0, original.Width, original.Height, GraphicsUnit.Pixel, attributes);
                return newBitmap;
            }
        }
    }
}
