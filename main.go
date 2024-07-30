package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"os"

	"gocv.io/x/gocv"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Println("Usage: go run main.go <input_image_path> <output_image_path>")
		return
	}

	imagePath := os.Args[1]
	outputPath := os.Args[2]

	err := processImage(imagePath, outputPath)
	if err != nil {
		fmt.Printf("Error processing image: %v\n", err)
	}
}

func processImage(imagePath string, outputPath string) error {
	src := gocv.IMRead(imagePath, gocv.IMReadColor)
	if src.Empty() {
		return fmt.Errorf("cannot open the image file: %s", imagePath)
	}
	defer src.Close()

	cascadeFile := "./haarcascade_frontalface_default.xml"
	cascade := gocv.NewCascadeClassifier()
	defer cascade.Close()

	if !cascade.Load(cascadeFile) {
		return fmt.Errorf("faild to load cascade file: %v", cascadeFile)
	}

	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(src, &gray, gocv.ColorBGRToGray)

	faces := cascade.DetectMultiScale(gray)
	fmt.Printf("Detected %d faces\n", len(faces))

	for _, face := range faces {
		expandedFace := image.Rect(
			max(0, face.Min.X-20),
			max(0, face.Min.Y-20),
			min(src.Cols(), face.Max.X+20),
			min(src.Rows(), face.Max.Y+20),
		)
		blurRegion(&src, expandedFace)
	}

	outFile, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer outFile.Close()

	img, err := src.ToImage()
	if err != nil {
		return fmt.Errorf("failed converting Mat to image: %v", err)
	}

	err = jpeg.Encode(outFile, img, nil)
	if err != nil {
		return fmt.Errorf("failed to encode output image: %v", err)
	}

	return nil
}

func blurRegion(mat *gocv.Mat, rect image.Rectangle) {
	if rect.Dx() <= 0 || rect.Dy() <= 0 {
		return
	}

	if rect.Min.X < 0 || rect.Min.Y < 0 || rect.Max.X > mat.Cols() || rect.Max.Y > mat.Rows() {
		return
	}

	faceRegion := mat.Region(rect)
	defer faceRegion.Close()

	kernelSize := image.Pt(101, 101)
	stdDev := 30.0
	gocv.GaussianBlur(faceRegion, &faceRegion, kernelSize, stdDev, stdDev, gocv.BorderDefault)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
