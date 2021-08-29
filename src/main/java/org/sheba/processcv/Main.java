/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.sheba.processcv;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import java.io.IOException;
import net.sourceforge.tess4j.Tesseract;
import org.im4java.core.ConvertCmd;
import org.im4java.core.IMOperation;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.core.Scalar;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.util.logging.Level;
import java.util.logging.Logger;
import net.sourceforge.tess4j.TesseractException;

/**
 *
 * @author kabir
 */
public class Main {

    public static void main(String[] args) throws IOException {

        String BASE_URL = "/home/kabir/Downloads/nid-images/";
        String TESS_DATA_PATH = "/usr/share/tesseract-ocr/4.00/tessdata/";
        boolean DEBUG = true;
//        System.loadLibrary(org.opencv.core.Core.NATIVE_LIBRARY_NAME);
        String inputFileName = "test20.jpg";
//        String inputFileName = "test2.jpg";
//        String inputFileName = "test3.jpg";
//        String inputFileName = "test8.jpg";

        System.loadLibrary("opencv_java420");
        Mat input = Imgcodecs.imread(BASE_URL + inputFileName);

        String fileName = inputFileName.substring(0, inputFileName.indexOf("."));
        //convert to pbm
        try {
            ConvertCmd cmd = new ConvertCmd();
            IMOperation op = new IMOperation();

            op.addImage(BASE_URL + inputFileName);
//            op.resize(1800, 2700);
//            op.resize(1961, 1231);
//            op.resize(width, height);
            op.addImage(BASE_URL + fileName + ".pbm");
            cmd.run(op);
        } catch (Exception ex) {
            //handle the exception
            ex.fillInStackTrace();
        }

        //read image
        Mat pbmImg = Imgcodecs.imread(BASE_URL + fileName + ".pbm");

        Imgproc.cvtColor(pbmImg, pbmImg, Imgproc.COLOR_BGR2GRAY);
        Core.bitwise_not(pbmImg, pbmImg);

        Mat horizontal = pbmImg.clone();
        Mat vertical = pbmImg.clone();

        int horizontalSize = 12;
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalSize, 1));

        Mat horizontalMask = new Mat();
        Mat horizontalErode = new Mat();
        Imgproc.dilate(horizontal, horizontalMask, kernel);
        Imgproc.erode(horizontalMask, horizontalErode, kernel);

        int verticalSize = 4;
        Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, verticalSize));

        Mat verticalMask = new Mat();
        Mat verticalErode = new Mat();
        Imgproc.dilate(vertical, verticalMask, kernel2);
        Imgproc.erode(verticalMask, verticalErode, kernel2);

        Mat merge = new Mat();
        Core.multiply(horizontalErode, verticalErode, merge);

        Mat kernel3 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(12, 1));

        Mat closing = new Mat();
        Imgproc.morphologyEx(merge, closing, Imgproc.MORPH_CLOSE, kernel3);

        Mat labels = new Mat();
        Mat stats = Mat.zeros(new Size(0, 0), 0);
        Mat centroids = Mat.zeros(new Size(0, 0), 0);

        Imgproc.connectedComponentsWithStats(closing, labels, stats, centroids);

        if (DEBUG) {
            Imgcodecs.imwrite(BASE_URL + fileName + "-shade.png", closing);
        }

        int count = 1;
        Mat newImg = new Mat(input.rows(), input.cols(), input.type(), new Scalar(255, 255, 255));
        for (int i = 1; i < stats.height(); ++i) {
            int left = (int) (stats.get(i, Imgproc.CC_STAT_LEFT)[0]);
            int top = (int) (stats.get(i, Imgproc.CC_STAT_TOP)[0]);
            int width = (int) (stats.get(i, Imgproc.CC_STAT_WIDTH)[0]);
            int height = (int) (stats.get(i, Imgproc.CC_STAT_HEIGHT)[0]);

            Point tl = new Point(left, top);
            Point br = new Point(left + width, top + height);
            Rect rect = new Rect(tl, br);

            if (height <= 8 || width <= 8) {
                continue;
            }

            if (height > 100 && width > 100) {
                continue;
            }
            Rect roi = new Rect(tl, br);
            Mat sub = input.submat(roi);
//            Imgcodecs.imwrite("/home/kabir/Downloads/nid-images/" + "partial-" + count + ".png", sub);
            sub.copyTo(newImg.submat(roi));

            count++;
        }
//        Imgcodecs.imwrite(BASE_URL + fileName + "-contours.png", newImg);

        Imgproc.cvtColor(newImg, newImg, Imgproc.COLOR_BGR2GRAY);
        if (DEBUG) {
            Imgcodecs.imwrite(BASE_URL + fileName + "-contours.png", newImg);
        }

        Mat result = new Mat();
//        Imgproc.adaptiveThreshold(newImg, result, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 30);
        Imgproc.threshold(newImg, result, 100, 255, Imgproc.THRESH_BINARY);
        if (DEBUG) {
            Imgcodecs.imwrite(BASE_URL + fileName + "-contours2.png", result);
        }

        try {
//            File image = new File(BASE_URL + fileName + "-contours2.png");

            // Create an empty image in matching format
            BufferedImage gray = new BufferedImage(result.width(), result.height(), BufferedImage.TYPE_BYTE_GRAY);

// Get the BufferedImage's backing array and copy the pixels directly into it
            byte[] data = ((DataBufferByte) gray.getRaster().getDataBuffer()).getData();
            result.get(0, 0, data);

            Tesseract tesseract = new Tesseract();
            tesseract.setDatapath(TESS_DATA_PATH);
            tesseract.setLanguage("ben+eng");
            tesseract.setPageSegMode(1);
            tesseract.setOcrEngineMode(1);

            String ocrResult = tesseract.doOCR(gray);
            System.out.println(ocrResult);
        } catch (TesseractException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
}
