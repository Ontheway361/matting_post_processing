#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <dirent.h>
#include <fstream>
#include <vector>
#include <string>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
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
    double threshold = trimap.rows * trimap.cols * 0.005;
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
    else if(count_255 > threshold){
        return "part sky";
    }
    return "not sky";
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
    resize(trimap, trimap, size);
    resize(alpha, alpha, size);
    return sky_mode;
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
    double channel_r=0, channel_g=0, channel_b=0;
    double channel_r_max=0, channel_g_max=0, channel_b_max=0;
    double channel_r_min=255, channel_g_min=255, channel_b_min=255;
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

int main(int argc, char **argv){
    
    string img_path     = argv[1];   // rgb_image list
    string prob_path    = argv[2];   // prob iamge list
    string alpha_path   = argv[3];   // matting_alpha path
    string concat_path  = argv[4];   // concat_img path

    
    ifstream img_file(img_path,ifstream::in);
    ifstream mask_file(prob_path,ifstream::in);
    
/*
    ofstream outfile;
    outfile.open(data_path,ios::app);
*/  
    int i = 0;
    if(img_file && mask_file){
        string img_str,mask_str;
        double total_time_cost = 0;
        while(getline(img_file,img_str) && getline(mask_file,mask_str)){
            i++;
            double start = static_cast<double>( getTickCount());
            string str_copy  = img_str;
            int idx = str_copy.rfind("/");
            string filename = str_copy.substr(idx+1);

            /*
            str_copy.erase(idx,1);
            int index = str_copy.rfind("/");
            string folder_name = str_copy.substr(index+1,idx-index-1);
				
            //vector<string> array = split(img_str,"/");
            if(folder_name.empty()){
                cerr << "can't find input folder...." << endl;
                continue;               
            }
            //string alpha_save_folder = alpha_path + folder_name;
            // test wether alpha_save_folder is exists...
            if( access(alpha_save_folder.c_str(),00) && mkdir(alpha_save_folder.c_str(),0777) ){
              cerr<<"creat "<<alpha_save_folder<<" failed...\n";
              return -1;
            }
            */

            cout << '\n' << i << ":\t" << filename;
            
            Mat img    = imread(img_str);
            Mat trimap = imread(mask_str, 0);
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
            Mat element1 = getStructuringElement(cv::MORPH_ELLIPSE, Size(8, 8)); //?
            dilate(trimap2, trimap2, element1);
            erode(trimap2, trimap2, element1);
*/
            Mat alpha;
            double eps = 0.02 * 0.02 * 255 * 255;
            string sky_mode = matting(img, trimap, alpha, 0.8, 1, 1, eps);
            if(sky_mode == "not sky") {
                Mat mask_0(trimap.size(),CV_8UC1,Scalar(0));
                alpha = mask_0;
                cout << "\tNot sky!\t";
            } else if(sky_mode == "all sky") {
                Mat mask_255(trimap.size(),CV_8UC1,Scalar(255));
                alpha = mask_255;
                cout << "\tAll sky!\t";
           }
           double single_time = (static_cast<double>( getTickCount() ) - start)/getTickFrequency();
           total_time_cost += single_time;
           cout << " cost time: " << single_time<< '\n';
            
           Mat concat_result;
           pasteMask(alpha, img, concat_result,0);
           imwrite(alpha_path + filename, alpha);
           imwrite(concat_path + filename, concat_result);
           img_str.clear();
           mask_str.clear();
        } // end for while
        cout<<"\ntotal_time_cost : "<<total_time_cost<<"\t; ave_cost_time : "<<total_time_cost/i<<endl;
    }// end for if
    else{
        cerr << "img_path or mask_path can't open..." << endl;
        return -1;
    }
    return 0;
}

