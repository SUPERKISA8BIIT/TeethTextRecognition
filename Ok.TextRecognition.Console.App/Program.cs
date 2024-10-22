using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text.RegularExpressions;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Numpy;
using Ok.TextRecognition.Detection;
using Ok.TextRecognition.Recognition;

namespace Ok.TextRecognition.App
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start detection");

            var size = new Size(1280*2, 1280*2);
            using var imageMat = new Mat("assets/photo_2024-10-21_23-02-17.jpg", loadType: ImreadModes.Color);
            var ratioX = imageMat.Width / (float)size.Width;
            var ratioY = imageMat.Height / (float)size.Height;

            using var resizedMat = new Mat();
            CvInvoke.Resize(imageMat, resizedMat, size);

            var detector = new EastTextDetector(width: size.Width, height: size.Height, confidenceThreshold: 0.8f, nmsThreshold: 0.8f);
            using var recognizer = new CrnnTextRecognizer();

            using var result = detector.DetectTexts(resizedMat);;

            if (result.ScoreText != null)
            {
                var render_img = result.ScoreText.copy();
                render_img = np.hstack(render_img, result.ScoreLink);
                using var ret_score_text = render_img.Cvt2HeatmapImg();
                render_img.Dispose();

                CvInvoke.Imwrite("result_1_masked.jpg", ret_score_text);
            }

            using var grayImage = new Mat();
            CvInvoke.CvtColor(imageMat, grayImage, ColorConversion.Bgr2Gray);

            int offsetY = 30;
            Dictionary<string, int> keyValuePairs = new();

            foreach (var idx in result.Boxes.Keys)
            {
                var points = result.Boxes[idx].Select(x => new PointF(x.X * ratioX, x.Y * ratioY)).ToArray();


                var text = recognizer.Recognize(grayImage, points);
                if (string.IsNullOrWhiteSpace(text)/* || !Regex.IsMatch(text, @"\d{2}[A-Za-z]")*/)
                {
                    continue;
                }
                if (!keyValuePairs.ContainsKey(text)) {
                    keyValuePairs[text] = 0;
                }
                keyValuePairs[text]++;
                CvInvoke.Polylines(imageMat, points.Select(pt => new Point((int)(pt.X), (int)(pt.Y))).ToArray(), true, new MCvScalar(255, 0, 0), thickness: 5);
            }

            foreach (var pair in keyValuePairs)
            {
                CvInvoke.PutText(imageMat, $"{pair.Key}: {pair.Value}", new Point(imageMat.Cols - 200, imageMat.Rows - 10 - offsetY), Emgu.CV.CvEnum.FontFace.HersheySimplex, 1.0, new MCvScalar(255, 255, 255), thickness: 2);
                offsetY += 40;
            }

            CvInvoke.Imwrite("result_1.jpg", imageMat);

        }
    }
}
