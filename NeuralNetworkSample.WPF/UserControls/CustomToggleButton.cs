using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

// ReSharper disable PossibleNullReferenceException

namespace NeuralNetworkSampleWPF.UserControls
{
    /// <summary>
    /// A simple styled toggle button
    /// </summary>
    public class CustomToggleButton : Button
    {
        public CustomToggleButton()
        {
            Toggled = false;
            MouseEnterOnTopBackground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF3E3E40"));
            MouseLeaveOnTopBackground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF1E1E1E"));
            this.Click += CustomToggleButton_Click;
            this.MouseEnter += onTopToggle_MouseEnter;
            this.MouseLeave += onTopToggle_MouseLeave;
        }

        private void CustomToggleButton_Click(object sender, RoutedEventArgs e) => Toggled = !Toggled;

        /// <summary>
        /// Indicates whether or not the toggle is activated
        /// </summary>
        private bool _PreventToggle;

        /// <summary>
        /// Gets or sets whether or not is it possible to toggle the button state
        /// </summary>
        public bool PreventToggle
        {
            get { return _PreventToggle; }
            set
            {
                if (value && !_PreventToggle)
                {
                    Click -= CustomToggleButton_Click;
                }
                else if (!value && _PreventToggle)
                {
                    Click += CustomToggleButton_Click;
                }
                _PreventToggle = value;
            }
        }

        /// <summary>
        /// Evento che si verifica quando viene cambiato lo stato del pulsante
        /// </summary>
        public event EventHandler OnToggled;

        /// <summary>
        /// Gets or sets the MouseOver background
        /// </summary>
        public Brush MouseEnterOnTopBackground { get; set; }

        /// <summary>
        /// Gets or sets the primary button background
        /// </summary>
        public Brush MouseLeaveOnTopBackground { get; set; }

        // Toggles the button background color on mouse over
        private void onTopToggle_MouseEnter(object sender, MouseEventArgs e) => Background = MouseEnterOnTopBackground;

        // Restores the primary background color
        private void onTopToggle_MouseLeave(object sender, MouseEventArgs e) => Background = MouseLeaveOnTopBackground;

        private bool _Toggled;

        /// <summary>
        /// Gets or sets the state of the toggle button
        /// </summary>
        public bool Toggled
        {
            get { return _Toggled; }
            set
            {
                if (_Toggled != value)
                {
                    _Toggled = value;
                    OnToggled?.Invoke(this, EventArgs.Empty);
                    if (value)
                    {
                        MouseEnterOnTopBackground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF0064A8"));
                        Background = MouseEnterOnTopBackground;
                        MouseLeaveOnTopBackground = MouseEnterOnTopBackground;
                    }
                    else
                    {
                        MouseEnterOnTopBackground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF3E3E40"));
                        Background = MouseEnterOnTopBackground;
                        MouseLeaveOnTopBackground = new SolidColorBrush((Color)ColorConverter.ConvertFromString("#FF1E1E1E"));
                    }
                }
            }
        }
    }
}
