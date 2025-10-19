using System;
using System.IO;
using OpenCvSharp;
using Tesseract;
using System.Drawing;
using OpenCvSharp.Extensions;


using Rect = OpenCvSharp.Rect;
using CvPoint = OpenCvSharp.Point;


class Program
{
    static void Main()
    {
        string imageFolder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", "..", "ScreenShot");
        string tessdataPath = @"C:\Program Files\Tesseract-OCR\tessdata"; // tessdataのパス


        foreach (var file in Directory.GetFiles(imageFolder, "*.png"))
        {
            using var full = new Mat(file, ImreadModes.Color);

            // 1. 外枠を除去
            var cropRect = new Rect(425, 126, full.Width - 425 - 84, full.Height - 126 - 65);
            var cropped = new Mat(full, cropRect);

            // 2. 横3分割
            int sliceWidth = cropped.Width / 3;
            for (int i = 0; i < 3; i++)
            {
                var sliceRect = new Rect(i * sliceWidth, 0, sliceWidth, cropped.Height);
                var slice = new Mat(cropped, sliceRect);

                // 🔸 このslice内で検出された数字領域を記録するリスト
                List<Rect> scannedNumberRegions = new List<Rect>();




                // 3. テンプレートマッチングで銀色コインの位置を検出
                var gray = new Mat();
                Cv2.CvtColor(slice, gray, ColorConversionCodes.BGR2GRAY);

                string[] templateFiles = Directory.GetFiles("Templates", "coin_template_*.png");


                foreach (var templatePath in templateFiles)
                {
                    var itemplate = Cv2.ImRead(templatePath, ImreadModes.Grayscale);
                    if (itemplate.Empty()) continue;
                    if (itemplate.Width > slice.Width || itemplate.Height > slice.Height)
                    {
                        Console.WriteLine($"テンプレート {templatePath} がスライスより大きいためスキップします。");
                        continue;
                    }

                    // 🔸 sliceの右半分だけを検索対象にする
                    var searchRegion = new Rect(slice.Width * 3 / 4, 0, slice.Width / 4, slice.Height);
                    var searchMat = new Mat(slice, searchRegion);

                    // 🔧 グレースケール化（テンプレートと形式を合わせる）
                    Cv2.CvtColor(searchMat, searchMat, ColorConversionCodes.BGR2GRAY);


                    var result = searchMat.MatchTemplate(itemplate, TemplateMatchModes.CCoeffNormed);

                    // 🔸 一致度が高い位置をすべて抽出
                    List<CvPoint> matchPoints = new List<CvPoint>();
                    for (int y = 0; y < result.Rows; y++)
                    {
                        for (int x = 0; x < result.Cols; x++)
                        {
                            float score = result.At<float>(y, x);
                            if (score >= 0.6)
                            {
                                matchPoints.Add(new CvPoint(x, y));
                            }
                        }
                    }

                    // 🔸 近接点をまとめる（重複除去）
                    List<CvPoint> filteredPoints = new List<CvPoint>();
                    int minDistance = 20;
                    foreach (var pt in matchPoints)
                    {
                        bool tooClose = filteredPoints.Any(p => Math.Abs(p.X - pt.X) < minDistance && Math.Abs(p.Y - pt.Y) < minDistance);
                        if (!tooClose)
                            filteredPoints.Add(pt);
                    }

                    // 🔸 検出済み領域を記録してスキップ
                    List<Rect> scannedRegions = new List<Rect>();

                    foreach (var loc in filteredPoints)
                    {
                        var absoluteLoc = new CvPoint(loc.X + searchRegion.X, loc.Y); // slice座標に戻す
                        var coinRect = new Rect(absoluteLoc.X, absoluteLoc.Y, itemplate.Width, itemplate.Height);

                        bool alreadyScanned = scannedRegions.Any(r => r.IntersectsWith(coinRect));
                        if (alreadyScanned)
                            continue;

                        scannedRegions.Add(coinRect);

                        // 🔸 数字領域（コインの右側）
                        var numberRect = new Rect(
                            coinRect.X + coinRect.Width,
                            coinRect.Y,
                            slice.Width - (coinRect.X + coinRect.Width),
                            30
                        );

                        if (numberRect.X < 0 || numberRect.Y < 0 || numberRect.X + numberRect.Width > slice.Width || numberRect.Y + numberRect.Height > slice.Height)
                            continue;

                        // 🔸 重複チェック
                        bool overlaps = scannedNumberRegions.Any(r => r.IntersectsWith(numberRect));
                        if (overlaps)
                            continue;

                        scannedNumberRegions.Add(numberRect);


                        string numberText = RunTesseractDigitsOnly(slice, numberRect, tessdataPath);

                        // 数字が検出された場合のみ名前を検索
                        if (!string.IsNullOrWhiteSpace(numberText) && numberText != "(なし)" && numberText != "(空画像)")
                        {
                            int nameY = coinRect.Y + 30 - 155;//上下位置
                            var nameRect = new Rect(0, nameY, 260, 30);
                            if (nameRect.X < 0 || nameRect.Y < 0 || nameRect.X + nameRect.Width > slice.Width || nameRect.Y + nameRect.Height > slice.Height)
                                continue;

                            string nameText = RunTesseract(slice, nameRect, tessdataPath);

                            Console.WriteLine($"数字: {numberText}, 名前: {nameText}");

                            Cv2.Rectangle(slice, coinRect, Scalar.Green, 2);
                            Cv2.Rectangle(slice, numberRect, Scalar.Red, 2);
                            Cv2.Rectangle(slice, nameRect, Scalar.Blue, 2);
                        }
                        else
                        {


                            Cv2.Rectangle(slice, coinRect, Scalar.Green, 2);
                            Cv2.Rectangle(slice, numberRect, Scalar.Red, 2);
                        }
                    }

                }

                string debugPath = Path.Combine("DebugImages", $"slice_{DateTime.Now:yyyyMMdd_HHmmssfff}.png");
                Directory.CreateDirectory("DebugImages");
                slice.SaveImage(debugPath);
                Console.WriteLine($"描画付きスライス画像を保存しました: {debugPath}");

            }
        }
    }

    static string RunTesseract(Mat image, Rect roi, string tessdataPath)
    {


        // ROI（指定範囲）を切り出し

        var cropped = new Mat(image, roi);
        if (cropped.Empty())
        {
            Console.WriteLine("切り出しに失敗しました（空の画像）。");
            return "(空画像)";
        }
        if (roi.X < 0 || roi.Y < 0 || roi.X + roi.Width > image.Width || roi.Y + roi.Height > image.Height)
        {
            Console.WriteLine($"無効なROI: x={roi.X}, y={roi.Y}, width={roi.Width}, height={roi.Height}");
            return "(領域外)";
        }


        // グレースケール化＋二値化（OCR精度向上）
        Cv2.CvtColor(cropped, cropped, ColorConversionCodes.BGR2GRAY);

        // 🔸 文字を認識するため拡大
        Cv2.Resize(cropped, cropped, new OpenCvSharp.Size(cropped.Width * 2.8, cropped.Height * 2.8), interpolation: InterpolationFlags.Linear);


        Cv2.GaussianBlur(cropped, cropped, new OpenCvSharp.Size(3, 3), 0);

        Cv2.Threshold(cropped, cropped, 0, 255, ThresholdTypes.Otsu);

        // Mat → Bitmap に変換
        using var bitmap = BitmapConverter.ToBitmap(cropped);

        // Bitmap → MemoryStream → byte[] → Pix に変換
        using var ms = new MemoryStream();
        bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
        ms.Position = 0;
        using var pix = Pix.LoadFromMemory(ms.ToArray());

        // OCRエンジンで読み取り
        using var engine = new TesseractEngine(tessdataPath, "jpn+eng", EngineMode.Default);
        using var page = engine.Process(pix);



        // 結果を返す
        return string.IsNullOrWhiteSpace(page.GetText()) ? "(なし)" : page.GetText().Trim();
    }


    static string RunTesseractDigitsOnly(Mat image, Rect roi, string tessdataPath)
    {
        if (roi.X < 0 || roi.Y < 0 || roi.X + roi.Width > image.Width || roi.Y + roi.Height > image.Height)
        {
            Console.WriteLine($"無効なROI: x={roi.X}, y={roi.Y}, width={roi.Width}, height={roi.Height}");
            return "(領域外)";
        }

        var cropped = new Mat(image, roi);
        if (cropped.Empty())
        {
            Console.WriteLine("切り出しに失敗しました（空の画像）。");
            return "(空画像)";
        }

        Cv2.CvtColor(cropped, cropped, ColorConversionCodes.BGR2GRAY);
        Cv2.GaussianBlur(cropped, cropped, new OpenCvSharp.Size(3, 3), 0);
        Cv2.Threshold(cropped, cropped, 0, 255, ThresholdTypes.Otsu);

        using var bitmap = BitmapConverter.ToBitmap(cropped);
        using var ms = new MemoryStream();
        bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
        ms.Position = 0;
        using var pix = Pix.LoadFromMemory(ms.ToArray());

        using var engine = new TesseractEngine(tessdataPath, "eng", EngineMode.Default);
        engine.SetVariable("tessedit_char_whitelist", "0123456789,"); // 🔸 数字とコンマだけ許可
        using var page = engine.Process(pix);

        return string.IsNullOrWhiteSpace(page.GetText()) ? "(なし)" : page.GetText().Trim();
    }

}