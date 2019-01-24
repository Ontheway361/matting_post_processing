#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include "global-matting/globalmatting.h"

using namespace cv;
using namespace std;




bool sroundedByWhite(Mat &mask, int c, int r, double rate, double count_prob){
    int R = mask.rows;
    int C = mask.cols;
    int len = (int)(rate * (R <= C ? R : C));
    len = max(1, len);

    int count = 0;

    int x1 = max(0, c-len);
    int x2 = min(mask.cols-1, c+len);
    int y1 = max(0, r-len);
    int y2 = min(mask.rows-1, r+len);

    Mat subRegion = mask(Rect(x1, y1, x2-x1, y2-y1));
    count = cv::countNonZero(subRegion);
    double threshold = count_prob*(x2-x1)*(y2-y1);
    return count >= max(1.0, threshold);
}

Mat getTrimap(Mat mask, Mat fore_probs){
    Mat trimap = Mat::zeros(mask.size(), mask.type());
    for (int r = 0; r < mask.rows; r++) {
        for (int c = 0; c < mask.cols; c++) {
            if((mask.at<uchar>(r, c) > 0) && ((fore_probs.at<uchar>(r, c) >= 0.8*255))){
                trimap.at<uchar>(r, c) = 255;
            } else if (sroundedByWhite(mask, c, r, 0.008, 0.01))
                trimap.at<uchar>(r, c) = 128;
            else {
                trimap.at<uchar>(r, c) = 0;
            }
        }
    }
    return trimap;
}

string is_sky(const Mat &trimap){
    int count_255 = 0;
    int count_0 = 0;
    double threshold = trimap.rows * trimap.cols * 0.01;
    for (int r = 0; r < trimap.rows; r++) {
        for (int c = 0; c < trimap.cols; c++) {
            if((trimap.at<uchar>(r, c) > 180)) {
                count_255++;
            }
            else if((trimap.at<uchar>(r, c) < 50)) {
                count_0 ++;
            }
        }
    }
    if(count_0 < threshold){
        return "all sky";
    }
    if(count_255 > threshold){
        //cout<<"count_255: " << count_255 << endl;
        return "part sky";
    }
    return "not sky";
}

void copy_file( const char* srce_file, const char* dest_file )
{
    std::ifstream srce( srce_file, std::ios::binary ) ;
    std::ofstream dest( dest_file, std::ios::binary ) ;
    dest << srce.rdbuf() ;
}

string matting(cv::InputArray _img, cv::InputArray _trimap, cv::OutputArray _alpha,
            double resize_rate=0.3, int expansion_rate=1, int filter_rate=3, double filter_eps=1e-10, int threshold=128) {
    cv::Size size = _img.size();
    Mat img = _img.getMat();
    Mat trimap = _trimap.getMat();
    resize(img, img, Size(static_cast<int>(round(resize_rate * img.cols)), static_cast<int>(round(resize_rate * img.rows))));
    resize(trimap, trimap, Size(static_cast<int>(round(resize_rate * trimap.cols)), static_cast<int>(round(resize_rate * trimap.rows))));

    Mat &alpha = _alpha.getMatRef();

    //expansionOfKnownRegions(img, trimap_refine, expansion_rate);
    string sky_mode = is_sky(trimap);
    if(sky_mode != "part sky")
      return sky_mode;
    cv::Mat foreground;
    globalMatting(img, trimap, foreground, alpha);
    cv::ximgproc::guidedFilter(img, alpha, alpha, filter_rate, filter_eps);
    
    for (int y = 0; y < trimap.rows; ++y) {
        for (int x = 0; x < trimap.cols; ++x) {
            if ((int) trimap.at<uchar>(y, x) == 0 || (int) alpha.at<uchar>(y, x) < threshold) {
                alpha.at<uchar>(y, x) = 0;
            } else if (trimap.at<uchar>(y, x) == 255 || alpha.at<uchar>(y, x) >= threshold) {
                alpha.at<uchar>(y, x) = 255;
            }
        }
    }
    //resize(trimap_refine, trimap_refine, size);
    resize(alpha, alpha, size);
    return sky_mode;
}

void img_combine(Mat img, Mat mask, Mat trimap_refine, Mat alpha,
                 string filename, bool show_img=false, string write_path="/Users/gaoyuan/CLionProjects/sky_matting/test_img/combine/tmp/EDSH1/"){

    int r = mask.rows;
    int c = mask.cols;
    cvtColor(alpha, alpha, CV_GRAY2BGR);
    cvtColor(trimap_refine, trimap_refine, CV_GRAY2BGR);
    cvtColor(mask, mask, CV_GRAY2BGR);

    int count=1;
    Mat combine = Mat::zeros(count*r, 4*c, img.type());
    Mat no1 = combine(Rect(0, 0, c, r));
    Mat no2 = combine(Rect(c, 0, c, r));
    Mat no3 = combine(Rect(2*c, 0, c, r));
    Mat no4 = combine(Rect(3*c, 0, c, r));

    img(cv::Rect(0, 0, c, r)).copyTo(no1);
    mask(cv::Rect(0, 0, c, r)).copyTo(no2);
    trimap_refine(cv::Rect(0, 0, c, r)).copyTo(no3);
    alpha(cv::Rect(0, 0, c, r)).copyTo(no4);

    if(show_img){
        imshow("combine", combine);
        waitKey();
    }
    if(!write_path.empty()){
        imwrite(write_path + filename, combine);
    }
}

void refine_trimap(cv::InputArray _img, cv::InputArray _trimap, cv::OutputArray _lap, cv::OutputArray _trimap_refine){
    Mat img = _img.getMat();
    Mat trimap = _trimap.getMat();
    Mat &lap = _lap.getMatRef();
    Mat &trimap_refine = _trimap_refine.getMatRef();

    Mat gray;
    cv::cvtColor(img, gray, COLOR_RGB2GRAY);
    cv::Laplacian(gray, lap, CV_64F);
    cv::convertScaleAbs(lap, lap);


    //统计trimap中前景区域对应img的rgb平均值
    int count_fore = 0;
    double channel_r=0, channel_g=0, channel_b=0, channel_r_max=0, channel_g_max=0, channel_b_max=0, channel_r_min=255, channel_g_min=255, channel_b_min=255;
    for (int r=0; r<img.rows; r++){
        for(int c=0; c<img.cols; c++){
            if(trimap.at<uchar>(r, c) == 255){
                Vec3b color = img.at<Vec3b>(r,c);
                channel_b += color[0];
                channel_g += color[1];
                channel_r += color[2];
                count_fore++;
            }
        }
    }
    channel_r /= count_fore;
    channel_g /= count_fore;
    channel_b /= count_fore;

    int add = 55;
    channel_r_max = min(255.0, channel_r + add);
    channel_g_max = min(255.0, channel_g + add);
    channel_b_max = min(255.0, channel_b + add);
    channel_r_min = max(0.0, channel_r - add);
    channel_g_min = max(0.0, channel_g - add);
    channel_b_min = max(0.0, channel_b - add);


    int count_trimap_add=0;

    trimap.copyTo(trimap_refine);
    for (int r = 0; r < gray.rows; r++) {
        for (int c = 0; c < gray.cols; c++) {
            Vec3b color = img.at<Vec3b>(r,c);
            int _bb = color[0];
            int _gg = color[1];
            int _rr = color[2];

            if (trimap_refine.at<uchar>(r,c) == 0 && //trimap当前像素为背景
                sroundedByWhite(trimap_refine, c, r, 0.05, 0.01) &&
                lap.at<uchar>(r,c) < 50 &&  //相随对应的灰度梯度<50
                _rr <= _gg && _gg <= _bb && //蓝色的rgb值有 r<g<b 的特点
                (_rr<=channel_r_max && _rr>=channel_r_min) && (_gg<=channel_g_max && _gg>=channel_g_min) && (_bb<=channel_b_max && _bb>=channel_b_min)) { //颜色在某个区间内

                trimap_refine.at<uchar>(r, c) = 128;
                count_trimap_add++;
            }
        }
    }

    cout << "count_trimap_add: " << count_trimap_add << endl;
    cout << "rgb: " << channel_b << ',' << channel_g << ',' << channel_r<< endl;
}

void pasteMask(Mat &mask, Mat &input, Mat &overlay, int threshold=0){
    if(mask.channels()>1){
        vector<Mat> channels(3);
        cv::split(mask, channels);
        mask = channels[0].clone();    
    }
    double alpha = 0.6;
    vector<int> gcolor = {180, 120, 200};
    overlay = input.clone();
    for(int i=0;i<mask.rows;i++){
        for(int j=0;j<mask.cols;j++){
            int value = mask.at<uchar>(i,j);
            if(value > threshold) {
                double thisalpha = double(value/255.0)*alpha;
                for(int k=0;k<3;k++)
                    overlay.at<Vec3b>(i,j).val[k]=(1-thisalpha)*overlay.at<Vec3b>(i,j).val[k] + thisalpha*gcolor[k];        
            }
        }
    }
}

int main(int argc, char **argv){

    string img_path     = argv[1];   // rgb_image path
    string trimap_path  = argv[2];   // prob iamge path
    string alpha_path   = argv[3];   // matting_alpha path

    DIR *dirp = opendir(img_path.c_str());
    struct dirent *directory;
    vector<string> fileVec;
    if(dirp == nullptr) {
        cerr << "the input path does not exist!" << endl;
        return -1;
    }
    while((directory = readdir(dirp)) != nullptr) {
        if ((directory->d_name[0] != '.')) {
            fileVec.push_back(directory->d_name);
        }
    }
    closedir(dirp);

    double total_time_cost = 0; 
    for(int i=0; i < fileVec.size(); i++) {

        double start = static_cast<double>( getTickCount());
        //string filename    = "4206386474-0009.jpg";
        //string img_path    = "/home/lujie/project/dataset/sky_20180910/batch20_npassimgs_img/";
        //string trimap_path = "/home/lujie/project/dataset/sky_20180910/batch20_npassimgs_trimap/";
        string filename = fileVec[i]; 
        cout << '\n' << i+1 << ":\t" << filename << endl;

        Mat img = imread(img_path + filename);
        Mat trimap = imread(trimap_path + filename, 0);
        
        resize(trimap, trimap, img.size());
/*
        Mat mask1 = prob.clone();
        for (int r = 0; r < prob.rows; r++) {
            for (int c = 0; c < prob.cols; c++) {
                if((mask1.at<uchar>(r, c) > 80)){
                    mask1.at<uchar>(r, c) = 255;
                }
                else {
                    mask1.at<uchar>(r, c) = 0;
                }
            }
        }

double start_trimap = static_cast<double>( getTickCount());
        //Trimap*******************************************************************************
        Mat trimap = getTrimap(mask1, prob);
double end_trimap = static_cast<double>( getTickCount());


        Mat trimap2, lap;
        refine_trimap(img, trimap, lap, trimap2);

        //膨胀&腐蚀*****************************************************************************
        Mat element1 = getStructuringElement(cv::MORPH_ELLIPSE, Size(8, 8)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
        dilate(trimap2, trimap2, element1);
        erode(trimap2, trimap2, element1);
*/
        //Matting******************************************************************************
        Mat alpha;
        double eps = 0.02 * 0.02 * 255 * 255;
        string sky_mode = matting(img, trimap, alpha, 0.8, 1, 1, eps);
        if(sky_mode == "not sky") {
            Mat mask_0(trimap.size(),CV_8UC1,Scalar(0));
            alpha = mask_0;
            cout << "Not sky!" << endl;
        } else if(sky_mode == "all sky") {
            Mat mask_255(trimap.size(),CV_8UC1,Scalar(255));
            alpha = mask_255;
            cout << "All sky!" << endl;
        }
        double single_time = (double)(getTickCount() - start)/getTickFrequency();
        total_time_cost += single_time;
        cout << "single time: " <<single_time<< '\n';

//        img_combine(img, mask1, trimap_refine, alpha, filename, false, "/Users/gaoyuan/CLionProjects/sky_matting/finaltest/combine/");
      /*
        string sky_img_srce = img_path + filename;
        string sky_img_dest = sky_img_path + filename;
        copy_file(sky_img_srce.c_str(), sky_img_dest.c_str());
        
        string sky_prob_srce = prob_path + filename;
        string sky_prob_dest = sky_prob_path + filename;
        copy_file(sky_prob_srce.c_str(), sky_prob_dest.c_str());
      */
        //imwrite(mask_path1+filename, mask1);
        //outfile<<img_path+filename+"  "+alpha_path+filename<<endl;
        //outfile<<(end_matting-start_matting)/getTickFrequency()<<endl;
        imwrite(alpha_path+filename, alpha);
        //Mat overlay;
        //pasteMask(alpha, img, overlay, 0);
        //imwrite("./alpha.jpg", alpha);
        //imwrite("./overlay.jpg", overlay);
    }
    cout<<"total_time_cost : "<<total_time_cost<<"\t"<<"ave_time_cost : "<<total_time_cost/fileVec.size();
    return 0;
}

