
#include "vo_features.h"

using namespace cv;
using namespace std;

#define MAX_FRAME 1000
#define MIN_NUM_FEAT 2000

//#define DEMO
#ifdef DEMO
char *POSES = "/home/raghav/Documents/poses/00.txt";
char *IMAGES = "/home/raghav/Documents/Dataset/00/image_2/";
#else
char *POSES = "/home/raghav/Documents/My_Dataset/Testdataset8/poses/R0.txt";
char *IMAGES = "/home/raghav/Documents/My_Dataset/Testdataset8/output/";
#endif

double getAbsoluteScale(int frame_id, int sequence_id, double z_cal) {

	string line;
	int i = 1;
	ifstream myfile(POSES);
	double x = 0, y = 0, z = 0;
	double x_prev, y_prev, z_prev;
	if (myfile.is_open()) {
		while ((getline(myfile, line)) && (i <= frame_id)) {
			z_prev = z;
			x_prev = x;
			y_prev = y;
			std::istringstream in(line);
			//cout << line << '\n';
			for (int j = 0; j < 12; j++) {
				in >> z;
				if (j == 7)
					y = z;
				if (j == 3)
					x = z;
			}

			i++;
		}
		myfile.close();
	}

	else {
		cout << "Unable to open file";
		return 0;
	}

	return sqrt(
			(x - x_prev) * (x - x_prev) + (y - y_prev) * (y - y_prev)
					+ (z - z_prev) * (z - z_prev));

}

int main(int argc, char **argv) {

	Mat img_1, img_2;
	Mat R_f, t_f; //the final rotation and tranlation vectors containing the

	ofstream myfile;
	myfile.open("results1_1.txt");

	double scale = 1.00;
	char filename1[200];
	char filename2[200];
#ifdef DEMO
	sprintf(filename1, "%s%06d.png", IMAGES, 0);
	sprintf(filename2, "%s%06d.png", IMAGES, 1);
#else
	sprintf(filename1, "%s%d.png", IMAGES, 1);
	sprintf(filename2, "%s%d.png", IMAGES, 2);
#endif

	char text[100];
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 1;
	int thickness = 1;
	cv::Point textOrg(10, 50);

	//read the first two frames from the dataset
	Mat img_1_c = imread(filename1);
	Mat img_2_c = imread(filename2);

	if (!img_1_c.data || !img_2_c.data) {
		std::cout << " --(!) Error reading images " << std::endl;
		return -1;
	}

	// we work with grayscale images
	cvtColor(img_1_c, img_1, COLOR_BGR2GRAY);
	cvtColor(img_2_c, img_2, COLOR_BGR2GRAY);

	// feature detection, tracking
	vector<Point2f> points1, points2; //vectors to store the coordinates of the feature points
	featureDetection(img_1, points1);        //detect features in img_1
	vector<uchar> status;
	featureTracking(img_1, img_2, points1, points2, status); //track those features to img_2

#ifdef DEMO
	double focal = 718.8560;
	cv::Point2d pp(607.1928, 185.2157);
#else
	double focal = 286.8511;
	cv::Point2d pp(419.4670, 386.9750);
#endif
	//recovering the pose and the essential matrix
	Mat E, R, t, mask;
	E = findEssentialMat(points2, points1, focal, pp, RANSAC, 0.999, 1.0, mask);
	recoverPose(E, points2, points1, R, t, focal, pp, mask);

	Mat prevImage = img_2;
	Mat currImage;
	vector<Point2f> prevFeatures = points2;
	vector<Point2f> currFeatures;

	char filename[100];

	R_f = R.clone();
	t_f = t.clone();

	clock_t begin = clock();

	namedWindow("Road facing camera", WINDOW_AUTOSIZE); // Create a window for display.
	namedWindow("Trajectory", WINDOW_AUTOSIZE); // Create a window for display.

	Mat traj = Mat::zeros(600, 600, CV_8UC3);

	for (int numFrame = 2; numFrame < MAX_FRAME; numFrame++) {
#ifdef DEMO
		sprintf(filename, "%s%06d.png", IMAGES, numFrame);
#else
		sprintf(filename, "%s%d.png", IMAGES, numFrame);
#endif

		Mat currImage_c = imread(filename);
		cvtColor(currImage_c, currImage, COLOR_BGR2GRAY);
		vector<uchar> status;
		featureTracking(prevImage, currImage, prevFeatures, currFeatures,
				status);

		E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC,
				0.999, 1.0, mask);
		recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

		Mat prevPts(2, prevFeatures.size(), CV_64F), currPts(2,
				currFeatures.size(), CV_64F);

		for (int i = 0; i < prevFeatures.size(); i++) { //this (x,y) combination makes sense as observed from the source code of triangulatePoints on GitHub
			prevPts.at<double>(0, i) = prevFeatures.at(i).x;
			prevPts.at<double>(1, i) = prevFeatures.at(i).y;

			currPts.at<double>(0, i) = currFeatures.at(i).x;
			currPts.at<double>(1, i) = currFeatures.at(i).y;
		}

		scale = getAbsoluteScale(numFrame, 0, t.at<double>(2));

		if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0))
				&& (t.at<double>(2) > t.at<double>(1))) {

			t_f = t_f + scale * (R_f * t);
			R_f = R * R_f;
		}
		// a redetection is triggered in case the number of feautres being trakced go below a particular threshold
		if (prevFeatures.size() < MIN_NUM_FEAT) {
			featureDetection(prevImage, prevFeatures);
			featureTracking(prevImage, currImage, prevFeatures, currFeatures,
					status);
		}
		prevImage = currImage.clone();
		prevFeatures = currFeatures;

		int x = int(t_f.at<double>(0)) + 300;
		int y = int(t_f.at<double>(2)) + 100;
		circle(traj, Point(x, y), 1, CV_RGB(255, 0, 0), 2);

		rectangle(traj, Point(10, 30), Point(550, 50), CV_RGB(0, 0, 0),
				CV_FILLED);
		sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm",
				t_f.at<double>(0), t_f.at<double>(1), t_f.at<double>(2));
		putText(traj, text, textOrg, fontFace, fontScale, Scalar::all(255),
				thickness, 8);

		imshow("Road facing camera", currImage_c);
		imshow("Trajectory", traj);

		waitKey(1);

	}

	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	cout << "Total time taken: " << elapsed_secs << "s" << endl;
	return 0;
}
