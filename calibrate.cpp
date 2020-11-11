/*!
 * @Author: Bin Jiang
 * @Date: 2020/10/5
 * @Description: The main function of the calibration code
 */
#include <iostream>
#include <dirent.h>
#include <sys/stat.h>
#include <fstream>
#include <unistd.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

#define ImgDataPath "/home/jiangbin/CLionProjects/Calibration/images/"  //The location where the image is stored.
#define DataXMLPath "/home/jiangbin/CLionProjects/Calibration/data.xml" //The location where the XML file is stored.
#define ImagesNum 50    //The number of images that used to calibrate
#define  w  6   //The number of black and white intersections of the checkerboard width
#define  h  7   //The number of black and white intersections of the checkerboard height

const  float chessboardSquareSize = 23.8f;  //The side length of each checkerboard square, in millimeters

/*!
 * @Input: None
 * @Output: None
 * @Description: This function is used to get images from camera and save them locally.
 */
void GetImage_from_Camera_and_Save(){
    VideoCapture cap(0);
    if(!cap.isOpened()){
        cout<<"Camera can't open!"<<endl;
        return ;
    }
    Mat frame;
    cap>>frame;
    system("/home/jiangbin/CLionProjects/Calibration/change.sh");
    for(int i =1;i<=ImagesNum;i++){
        cap>>frame;
        waitKey(5);
    }
    if (access(ImgDataPath, 0) == -1){
        mkdir(ImgDataPath,S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }
    for(int i =1;i<=ImagesNum;i++){
        Mat DoubleImg;
        cap >> frame;
        if(frame.empty()) {
            break;
        }
        //imshow("FRAME_raw",frame);
        resize(frame,DoubleImg,Size(640,240),(0,0),(0,0),CV_INTER_AREA);
        //imshow("Double_img",DoubleImg);
        Mat LeftImg = DoubleImg(Rect(0,0,320,240));
        Mat RightImg = DoubleImg(Rect(320,0,320,240));
        imshow("LeftImg",LeftImg);
        imshow("RightImg",RightImg);
        imwrite(ImgDataPath + to_string(i) + "_Left.jpg",LeftImg);
        imwrite(ImgDataPath + to_string(i) + "_Right.jpg",RightImg);
        waitKey(300);
    }
}

/*!
 * @Input: None
 * @Output: None
 * @Description: This function is used to write the absolute path of the images into the XML file.
 */
void ImageFilesPath_to_XML(){
    DIR * dir;
    struct dirent ** ptr;
    string rootdirPath = ImgDataPath;
    string x,dirPath;
    dir = opendir((char *)rootdirPath.c_str());
    ofstream fileout(DataXMLPath,ios::trunc);
    std::ofstream	OsWrite(DataXMLPath,std::ofstream::app);
    OsWrite<<"<opencv_storage>";
    OsWrite<<endl;
    OsWrite<<"<images>";
    OsWrite<<endl;
    OsWrite<<endl;
    int n;
    n = scandir(rootdirPath.c_str(), &ptr, 0, alphasort);
    if(n < 0) {
        cout << "scandir return " << n << endl;
    }else{
        int index=0;
        while(index < n)
        {
            x=ptr[index]->d_name;
            if(x[0] != '..' && x[0] != '.'){
                dirPath = rootdirPath + x;
                //OsWrite<<"\"";
                OsWrite<<dirPath;
                //OsWrite<<"\"";
                OsWrite<<endl;
            }
            free(ptr[index]);
            index++;
        }
        free(ptr);
    }
    OsWrite<<endl;
    OsWrite<<"</images>";
    OsWrite<<endl;
    OsWrite<<"</opencv_storage>";
    OsWrite.close();
    closedir(dir);
}

/*!
 * @Input: const string& filename: Name of the XML file.
 *         vector<string>& list: The absolute path list of the images.
 * @Output: Boolean
 * @Description: This function is used to list the absolute path list of the images.
 */
static bool readStringList(const string& filename, vector<string>& list)
{
    list.resize(0);
    fstream f(filename);
    if(!f.is_open()){
        cout<<"fs can't open!"<<endl;
        return false;
    }
    string line;
    while (getline(f, line)) {
        if (line[0] == '\/'){
            list.push_back(line);
            //cout<<line<<endl;
        }else
            continue;
    }
    return true;
}

/*!
 * @Input: const Size& boardSize: The size of the chess board
 *         float squareSize: The length of the checkerboard
 *         vector<Point3f>& corners: Three-dimensional point vector of the chess board corners
 * @Output: None
 * @Description: This function is used to calculate the chessboard corners in each image.
 */
static void calcChessboardCorners(const Size& boardSize, float squareSize, vector<Point3f>& corners)
{
    corners.resize((float)0);
    for (int i = 0; i < boardSize.height; i++)
        for (int j = 0; j < boardSize.width; j++)
        {
            // calculate the corners' XYZ and push them into the three-dimensional point vector
            corners.emplace_back((float)j*squareSize, (float)i*squareSize, (float)0);
        }
}

/*!
 * @Input: Mat& intrMat: 
 *         Mat& distCoeffs:
 *         vector<vector<Point2f>>& imagePoints:
 *         vector<vector<Point3f>>& ObjectPoints:
 *         Size& imageSize:
 *         const int cameraId:
 *         vector<string> imageList:
 *  @Output: Boolean
 *  @Description: This function is used to calibrate the camera.
 */
bool calibrate(Mat& intrMat, Mat& distCoeffs, vector<vector<Point2f>>& imagePoints,
               vector<vector<Point3f>>& ObjectPoints, Size& imageSize, const int cameraId,
               vector<string> imageList)
{
    double rms = 0; //Reprojection error

    Size boardSize; //Opencv's Size class, describing the width and height
    boardSize.width = w;
    boardSize.height = h;

    vector<Point2f> pointBuf;
    float squareSize = chessboardSquareSize;

    vector<Mat> rvecs, tvecs;   //Define the rotation matrix and translation vector of the two cameras

    bool ok = false;

    int nImages = (int)imageList.size() / 2;
    cout <<"The number of images: "<< nImages;
    namedWindow("View", 1);

    int nums = 0;   //Number of effective checkers

    for (int i = 0; i< nImages; i++)
    {
        Mat view, viewGray;
        cout<<"Now: "<<imageList[i * 2 + cameraId]<<endl;
        view = imread(imageList[i * 2 + cameraId]);
        imshow("image",view);
        imageSize = view.size();
        cvtColor(view, viewGray, COLOR_BGR2GRAY);

        bool found = findChessboardCorners(view, boardSize, pointBuf,
                                           CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);//寻找棋盘格角点
        if (found)
        {
            nums++;
            cornerSubPix(viewGray, pointBuf, Size(11, 11),
                         Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
            drawChessboardCorners(view, boardSize, Mat(pointBuf), found);
            bitwise_not(view, view);
            imagePoints.push_back(pointBuf);
            cout << '.';
        }
        else{
            cout<<"Wrong"<<endl;
        }
        imshow("View", view);
        waitKey(100);
    }

    cout << "Number of valid checkerboard pictures: " << nums << endl;

    //calculate chessboardCorners
    calcChessboardCorners(boardSize, squareSize, ObjectPoints[0]);
    ObjectPoints.resize(imagePoints.size(), ObjectPoints[0]);

    rms = calibrateCamera(ObjectPoints, imagePoints, imageSize, intrMat, distCoeffs,
                          rvecs, tvecs);
    ok = checkRange(intrMat) && checkRange(distCoeffs);

    if (ok)
    {
        cout << "Done with RMS error = " << rms << endl;
        return true;
    }
    else
        return false;
}

/*!
 * Input: None
 * Output: None
 * Description: This function is used to operate the calibration.
 */
void Start_Calibrate(){
    //initialize some parameters
    bool okcalib = false;
    Mat intrMatFirst, intrMatSec, distCoeffsFirst, distCoffesSec;
    Mat R, T, E, F, RFirst, RSec, PFirst, PSec, Q;
    vector<vector<Point2f>> imagePointsFirst, imagePointsSec;
    vector<vector<Point3f>> ObjectPoints(1);
    Rect validRoi[2];
    Size imageSize;
    int cameraIdFirst = 0, cameraIdSec = 1;
    double rms = 0;

    //get pictures and calibrate
    vector<string> imageList;
    string filename = DataXMLPath;
    bool okread = readStringList(filename, imageList);
    if (!okread || imageList.empty())
    {
        cout << "can not open " << filename << " or the string list is empty" << endl;
        return ;
    }
    if (imageList.size() % 2 != 0)
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    FileStorage fs("/home/jiangbin/CLionProjects/Calibration/intrinsics.yml", FileStorage::WRITE);
    //calibrate

    cout << "calibrate left camera..." << endl;
    okcalib = calibrate(intrMatFirst, distCoeffsFirst, imagePointsFirst, ObjectPoints,
                        imageSize, cameraIdFirst, imageList);

    if (!okcalib)
    {
        cout << "fail to calibrate left camera" << endl;
        return;
    }
    else
    {
        cout << "calibrate the right camera..." << endl;
    }


    okcalib = calibrate(intrMatSec, distCoffesSec, imagePointsSec, ObjectPoints,
                        imageSize, cameraIdSec, imageList);

    fs << "M1" << intrMatFirst << "D1" << distCoeffsFirst <<
       "M2" << intrMatSec << "D2" << distCoffesSec;

    if (!okcalib)
    {
        cout << "fail to calibrate the right camera" << endl;
        return;
    }
    destroyAllWindows();

    //estimate position and orientation
    cout << "estimate position and orientation of the second camera" << endl
         << "relative to the first camera..." << endl;
    cout << "intrMatFirst:";
    cout << intrMatFirst << endl;
    cout << "distCoeffsFirst:";
    cout << distCoeffsFirst << endl;
    cout << "intrMatSec:";
    cout << intrMatSec << endl;
    cout << "distCoffesSec:";
    cout << distCoffesSec << endl;

    rms = stereoCalibrate(ObjectPoints, imagePointsFirst, imagePointsSec,
                          intrMatFirst, distCoeffsFirst, intrMatSec, distCoffesSec,
                          imageSize, R, T, E, F, CALIB_USE_INTRINSIC_GUESS,//CV_CALIB_FIX_INTRINSIC,
                          TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 1e-6));          //计算重投影误差
    cout << "done with RMS error=" << rms << endl;

    //stereo rectify
    cout << "stereo rectify..." << endl;
    stereoRectify(intrMatFirst, distCoeffsFirst, intrMatSec, distCoffesSec, imageSize, R, T, RFirst,
                  RSec, PFirst, PSec, Q, CALIB_ZERO_DISPARITY, -1, imageSize, &validRoi[0], &validRoi[1]);
    cout << "Q" << Q << endl;
    cout << "P1" << PFirst << endl;
    cout << "P2" << PSec << endl;
    //read pictures for 3d-reconstruction

    if (fs.isOpened())
    {
        cout << "in";
        fs << "R" << R << "T" << T << "R1" << RFirst << "R2" << RSec << "P1" << PFirst << "P2" << PSec << "Q" << Q;
        fs.release();
    }

    namedWindow("canvas", 1);
    cout << "read the picture for 3d-reconstruction..."<<endl;;
    Mat canvas(imageSize.height, imageSize.width * 2, CV_8UC3), viewLeft, viewRight;
    Mat canLeft = canvas(Rect(0, 0, imageSize.width, imageSize.height));
    Mat canRight = canvas(Rect(imageSize.width, 0, imageSize.width, imageSize.height));

    viewLeft = imread(imageList[6], 1);//cameraIdFirst
    viewRight = imread(imageList[7], 1); //cameraIdSec
    cout<<"Choose: "<<imageList[6]<<"  "<<imageList[7]<<endl;
    viewLeft.copyTo(canLeft);
    viewRight.copyTo(canRight);
    cout << "done" << endl;
    imshow("canvas", canvas);
    waitKey(1500);


    //stereoRectify
    Mat rmapFirst[2], rmapSec[2], rviewFirst, rviewSec;
    initUndistortRectifyMap(intrMatFirst, distCoeffsFirst, RFirst, PFirst,
                            imageSize, CV_16SC2, rmapFirst[0], rmapFirst[1]);//CV_16SC2
    initUndistortRectifyMap(intrMatSec, distCoffesSec, RSec, PSec,//CV_16SC2
                            imageSize, CV_16SC2, rmapSec[0], rmapSec[1]);
    remap(viewLeft, rviewFirst, rmapFirst[0], rmapFirst[1], INTER_LINEAR);
    imshow("remap", rviewFirst);
    waitKey(2000);

    remap(viewRight, rviewSec, rmapSec[0], rmapSec[1], INTER_LINEAR);
    rviewFirst.copyTo(canLeft);
    rviewSec.copyTo(canRight);

    //rectangle(canLeft, validRoi[0], Scalar(255, 0, 0), 3, 8);
    //rectangle(canRight, validRoi[1], Scalar(255, 0, 0), 3, 8);

    Mat before_rectify = imread("/home/jiangbin/CLionProjects/Calibration/images/1_Right.jpg");

    for (int j = 0; j <= canvas.rows; j += 16)
        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);

    for (int j = 0; j <= canvas.rows; j += 16)
        line(before_rectify, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
    cout << "stereo rectify done" << endl;

    imshow("Before", before_rectify);
    imshow("After", canvas);

    waitKey(400000);
}

/*!
 * @Input: None
 * @Output: None
 * @Description: The main function
 */
int main() {
    GetImage_from_Camera_and_Save();

    ImageFilesPath_to_XML();

    Start_Calibrate();

    return 0;
}
