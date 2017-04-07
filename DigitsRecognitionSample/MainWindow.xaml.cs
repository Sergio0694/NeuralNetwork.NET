using System;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using DigitsRecognitionSample.Views;

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

        private void Button_Click_1(object sender, RoutedEventArgs e) => NavigatePage<XYWindow>();

        private void Button_Click_2(object sender, RoutedEventArgs e) => NavigatePage<DigitsWindow>();

        /// <summary>
        /// Manages the navigation by making sure no double instances are created
        /// </summary>
        /// <typeparam name="T">The page to open</typeparam>
        private void NavigatePage<T>() where T : Window, new()
        {
            Window window = Application.Current.Windows.OfType<Window>().SingleOrDefault(x => x.GetType() == typeof(T));
            if (window == null)
            {
                new T { Left = Left + 25, Top = Top + 25 }.Show();
            }
            else window.Focus();
        }

        #endregion
    }
}
