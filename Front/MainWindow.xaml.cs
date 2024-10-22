using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Ok.TextRecognition.Detection;
using Ok.TextRecognition.Recognition;

namespace Front
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private async void InputFile_Click(object sender, System.Windows.RoutedEventArgs e)
        {
            try
            {
                var openFileDialog = new Microsoft.Win32.OpenFileDialog();
                openFileDialog.Filter = "Image files (*.jpg) | *.jpg";
                if (openFileDialog.ShowDialog() == true)
                {

                    var filePath = openFileDialog.FileName;
                    OpenFileBTN.Tag = filePath;

                    await Task.Run(() => ProduceImage(filePath));
                }
            }
            catch (Exception ex)
            {
                System.Windows.MessageBox.Show($"Error: {ex.Message}");
            }
            finally
            {
                Loader.Visibility = System.Windows.Visibility.Collapsed;
                LoaderText.Visibility = System.Windows.Visibility.Collapsed;
            }
        }

        private void ProduceImage(string filename)
        {
            Dispatcher.Invoke(() =>
            {
                Loader.Visibility = System.Windows.Visibility.Visible;
                LoaderText.Visibility = System.Windows.Visibility.Visible;
                SecondImage.Source = null;
            });

            var size = new System.Drawing.Size(1280 * 2, 1280 * 2);
            using var imageMat = new Mat(filename, loadType: ImreadModes.Color);
            var ratioX = imageMat.Width / (float)size.Width;
            var ratioY = imageMat.Height / (float)size.Height;
            using var resizedMat = new Mat();
            CvInvoke.Resize(imageMat, resizedMat, size);
            var detector = new EastTextDetector(width: size.Width, height: size.Height, confidenceThreshold: 0.4f, nmsThreshold: 0.4f);
            using var recognizer = new CrnnTextRecognizer();
            using var result = detector.DetectTexts(resizedMat);
            using var grayImage = new Mat();
            CvInvoke.CvtColor(imageMat, grayImage, ColorConversion.Bgr2Gray);
            Dictionary<string, int> keyValuePairs = new();

            List<(string text, PointF pos, PointF[] box)> LetterDigit = new();
            List<(string text, PointF pos, PointF[] box)> DigitLetter = new();

            foreach (var idx in result.Boxes.Keys)
            {
                var points = result.Boxes[idx].Select(x => new PointF(x.X * ratioX, x.Y * ratioY)).ToArray();
                var text = recognizer.Recognize(grayImage, points);

                var isDigitLetterLetter = Regex.IsMatch(text, @"\d{1}[A-Za-z]{2}");
                var isCorrectLength = text.Length == 3;
                if (isDigitLetterLetter)
                {
                    text = text.Replace('O', '0');
                }

                if (string.IsNullOrWhiteSpace(text))
                {
                    continue;
                }
                var avgPoint = new PointF(points.Average(x => x.X), points.Average(x => x.Y));

                if (Regex.IsMatch(text, @"\D{1}\d+"))
                {
                    LetterDigit.Add(new(text, avgPoint, points));
                } else if (Regex.IsMatch(text, @"\d{2}\D{1}"))
                {
                    var rigthBox = GetAverageColor(imageMat, new Rectangle(Math.Min(imageMat.Width - 5, (int)avgPoint.X + 60), Math.Min((int)avgPoint.Y, imageMat.Height - 5), 5, 5));

                    var isWhiteBackground = IsCloseToWhite(rigthBox);
                    //CvInvoke.DrawMarker(imageMat, new Point(Math.Min(imageMat.Width - 5, (int)avgPoint.X + 60), Math.Min((int)avgPoint.Y, imageMat.Height - 5)), new MCvScalar(0, 255, 0), MarkerTypes.Cross);

                    if (!isWhiteBackground) continue;

                    DigitLetter.Add(new(text, avgPoint, points));
                }
            }

            var sectorSize = new System.Drawing.Size(200, 100);
            var sectorAngle = -20d; // Set only negative angle

            foreach (var item in DigitLetter)
            {
                CvInvoke.Polylines(imageMat, item.box.Select(pt => new System.Drawing.Point((int)(pt.X), (int)(pt.Y))).ToArray(), true, new MCvScalar(0, 255, 0), thickness: 5);
                var leftLetterDigit = LetterDigit.Where(x => IsPointInSector(item.pos.ToPoint(), sectorSize, 180 + sectorAngle, 180 - sectorAngle, x.pos.ToPoint()));
                var nearestLetterDigit = leftLetterDigit.OrderBy(x => MathF.Sqrt(MathF.Pow(x.pos.X - item.pos.X, 2) + MathF.Pow(x.pos.Y - item.pos.Y, 2))).FirstOrDefault();

                string text;
                if (nearestLetterDigit != default)
                {
                   // DrawSector(imageMat, item.pos.ToPoint(), sectorSize, 180 + sectorAngle, 180 - sectorAngle, new MCvScalar(0, 0, 0), 1);
                   // CvInvoke.Line(imageMat, item.pos.ToPoint(), nearestLetterDigit.pos.ToPoint(), new MCvScalar(0, 0, 0));
                    CvInvoke.Polylines(imageMat, nearestLetterDigit.box.Select(pt => new System.Drawing.Point((int)(pt.X), (int)(pt.Y))).ToArray(), true, new MCvScalar(255, 0, 0), thickness: 2);
                    text = $"{nearestLetterDigit.text}-{item.text}";
                } else
                {
                    text = item.text;
                }

                if (!keyValuePairs.ContainsKey(text))
                {
                    keyValuePairs[text] = 0;
                }
                keyValuePairs[text]++;
            }

            double fontScale = 1.5;
            int thickness = 4;
            int baseline = 0;
            int offsetY = 0;
            int maxTextWidth = 0;
            int totalTextHeight = 0;

            foreach (var pair in keyValuePairs)
            {

                string text = $"{pair.Key}: {pair.Value}";

                System.Drawing.Size textSize = CvInvoke.GetTextSize(text,
                    Emgu.CV.CvEnum.FontFace.HersheySimplex,
                    fontScale,
                    thickness,
                    ref baseline);

                if (textSize.Width > maxTextWidth)
                {
                    maxTextWidth = textSize.Width;
                }

                totalTextHeight += textSize.Height + 10;
            }

            var startTextPosition = new System.Drawing.Point(imageMat.Cols - 300, imageMat.Rows - totalTextHeight - 20);

            var rectTopLeft = new System.Drawing.Point(startTextPosition.X, startTextPosition.Y - 10);
            var rectBottomRight = new System.Drawing.Point(startTextPosition.X + maxTextWidth, startTextPosition.Y + totalTextHeight);

            CvInvoke.Rectangle(imageMat, new System.Drawing.Rectangle(rectTopLeft, new System.Drawing.Size(maxTextWidth, totalTextHeight + 10)), new MCvScalar(0, 0, 0), -1);

            offsetY = 0;
            foreach (var pair in keyValuePairs)
            {
                string text = $"{pair.Key}: {pair.Value}";

                var textPosition = new System.Drawing.Point(imageMat.Cols - 300, startTextPosition.Y + offsetY + 30);

                CvInvoke.PutText(imageMat, text, textPosition,
                    Emgu.CV.CvEnum.FontFace.HersheySimplex,
                    fontScale,
                    new MCvScalar(0, 255, 0),
                    thickness);

                offsetY += CvInvoke.GetTextSize(text, Emgu.CV.CvEnum.FontFace.HersheySimplex, fontScale, thickness, ref baseline).Height + 10;
            }

            var tempFilePath = System.IO.Path.GetTempFileName() + ".jpg";
            CvInvoke.Imwrite(tempFilePath, imageMat);

            Dispatcher.Invoke(() =>
            {
                SecondImage.Source = new BitmapImage(new Uri(tempFilePath));
            });
        }

        public void DrawSector(Mat image, System.Drawing.Point center, System.Drawing.Size axes, double startAngle, double endAngle, MCvScalar color, int thickness)
        {
            CvInvoke.Ellipse(image, center, axes, 0, startAngle, endAngle, color, thickness);
            
            double startAngleRad = startAngle * Math.PI / 180.0;
            double endAngleRad = endAngle * Math.PI / 180.0;

            System.Drawing.Point startPoint = new(
                center.X + (int)(axes.Width * Math.Cos(startAngleRad)),
                center.Y - (int)(axes.Height * Math.Sin(startAngleRad))
            );

            System.Drawing.Point endPoint = new(
                center.X + (int)(axes.Width * Math.Cos(endAngleRad)),
                center.Y - (int)(axes.Height * Math.Sin(endAngleRad))
            );
            
            CvInvoke.Line(image, center, startPoint, color, thickness);
            CvInvoke.Line(image, center, endPoint, color, thickness);
        }

        public bool IsPointInSector(System.Drawing.Point center, System.Drawing.Size axes, double startAngle, double endAngle, System.Drawing.Point point)
        {
            double startAngleRad = startAngle * Math.PI / 180.0;
            double endAngleRad = endAngle * Math.PI / 180.0;
            
            double dx = point.X - center.X;
            double dy = center.Y - point.Y; // Инвертируем Y, так как ось Y направлена вниз

            // Преобразование координат в систему координат эллипса
            double normalizedX = dx / axes.Width;
            double normalizedY = dy / axes.Height;

            // Вычисление угла и расстояния от центра до точки в нормализованной системе координат
            double angle = Math.Atan2(normalizedY, normalizedX);
            if (angle < 0) angle += 2 * Math.PI; // Приведение угла к диапазону [0, 2π]
            double distance = Math.Sqrt(normalizedX * normalizedX + normalizedY * normalizedY);

            // Проверка, находится ли точка в пределах углов сектора
            bool isWithinAngles = (startAngleRad <= angle && angle <= endAngleRad) ||
                                  (startAngleRad > endAngleRad && (angle >= startAngleRad || angle <= endAngleRad));

            // Проверка, находится ли точка в пределах радиусов сектора
            bool isWithinRadius = distance <= 1.0;

            return isWithinAngles && isWithinRadius;
        }

        public static System.Drawing.Color GetAverageColor(Mat image, Rectangle region)
        {
            Mat croppedMat = new(image, region);
            
            Image<Bgr, Byte> croppedImg = croppedMat.ToImage<Bgr, byte>();

            byte[,,] pixels = croppedImg.Data;
            
            double bSum = 0, gSum = 0, rSum = 0;
            int pixelCount = pixels.GetLength(0) * pixels.GetLength(1);

            for (int i = 0; i < pixels.GetLength(0); i++)
            {
                for (int j = 0; j < pixels.GetLength(1); j++)
                {
                    bSum += pixels[i, j, 0];
                    gSum += pixels[i, j, 1];
                    rSum += pixels[i, j, 2];
                }
            }

            byte avgB = (byte)(bSum / pixelCount);
            byte avgG = (byte)(gSum / pixelCount);
            byte avgR = (byte)(rSum / pixelCount);

            // Возврат среднего цвета
            return System.Drawing.Color.FromArgb(avgR, avgG, avgB);

        }

        public bool IsCloseToWhite(Color color, int threshold = 30)
        {
            // Пороговое значение определяет, насколько близко цвет должен быть к белому.
            // 0 — это точное совпадение с белым, и чем выше значение, тем больший разброс допускается.
            int rDiff = Math.Abs(127 - color.R);
            int gDiff = Math.Abs(127 - color.G);
            int bDiff = Math.Abs(127 - color.B);

            return rDiff <= threshold && gDiff <= threshold && bDiff <= threshold;

        }
    }
}
