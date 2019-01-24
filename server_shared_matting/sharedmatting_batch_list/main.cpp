#include <iostream>
#include <dirent.h>
#include <fstream>
#include <string>
#include <time.h>
#include "SharedMatting.h"

using namespace std;

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

bool isUnknownrows(cv::Mat &trimap, int y){
    int count_128 = 0;
    float ratio = 0.9;
    for(int i = 0 ; i < trimap.cols; i++)
        if(trimap.at<uchar>(y,i) == 128){
            count_128++;
        }
    return count_128 > (int)(ratio * trimap.cols);
}

int  isUnknowncols(cv::Mat &trimap, int x){
    int count_128 = 0, count_255 = 0;
    int idx = 0;
    float ratio = 0.6;
    for(int y = 10; y < trimap.rows; y++){
        idx = y;
        if(trimap.at<uchar>(y,x) > 128)
            count_255++;
        else if(trimap.at<uchar>(y,x) == 128)
            count_128++;
        else
            break;
    }
    if(count_128 > (int)(ratio * idx))
        return idx - 6;
    else
        return 0;
}

void refineTrimap(Mat &trimap){
    // trans trimap to F,U,B
    int high_threshold = 180, low_threshold = 50;
    for(int y = 0; y < trimap.rows; y++){
        for(int x = 0; x < trimap.cols; x++){
            if(trimap.at<uchar>(y,x) > high_threshold)
                trimap.at<uchar>(y,x) = 255;
            else if(trimap.at<uchar>(y,x) < low_threshold)
                trimap.at<uchar>(y,x) = 0;
            else
                trimap.at<uchar>(y,x) = 128;
        }
    }

    // delete up-boundary
    for(int y = 0; y < trimap.rows; y++){
        bool flag = isUnknownrows(trimap, y);
        if(flag){
            for(int x = 0; x < trimap.cols; x++)
                trimap.at<uchar>(y,x) = 255;
        }else{
            break;
        }
    }
    int gray_band = 10;

    // delete left-boundary
    for(int x = 0; x < gray_band; x++){
        int idx = isUnknowncols(trimap, x);
        for(int y = 0; y < idx; y++)
            trimap.at<uchar>(y,x) = 255;
    }

    // delete right-boundary
    for(int x = trimap.cols-gray_band; x < trimap.cols; x++){
        int idx = isUnknowncols(trimap, x);
        for(int y = 0; y < idx; y++)
            trimap.at<uchar>(y,x) = 255;
    }
}

int is_sky(const cv::Mat &trimap){
    int count_255 = 0;
    int count_0 = 0;
    double threshold = trimap.rows * trimap.cols * 0.005;
    for (int r = 0; r < trimap.rows; r++) {
        for (int c = 0; c < trimap.cols; c++) {
            if((trimap.at<uchar>(r, c) == 255))
                count_255++;
            else if((trimap.at<uchar>(r, c) == 0))
                 count_0 ++;    
        }
            
    }
    if(count_0 < threshold)
        return 1;
    else if(count_255 > threshold)
        return 2;
    else        
        return 0;
}

int main(int argc, const char * argv[]) {

    string img_path     = argv[1];   // rgb_image list
    string trimap_path  = argv[2];   // prob iamge list
    string alpha_path   = argv[3];   // matting_alpha path
    string concat_path  = argv[4];   // concat_result path

    ifstream img_file(img_path,ifstream::in);
    ifstream trimap_file(trimap_path,ifstream::in);

    int i = 0;
    if(img_file && trimap_file){
        string img_str, tri_str;
        double total_time_cost = 0;
        while(getline(img_file, img_str) && getline(trimap_file, tri_str)){
            i++;
            int idx = img_str.rfind("/");
            string filename = img_str.substr(idx+1);

            cout << '\n' << i << ":\t" << filename;

            Mat img    = imread(img_str);
            Mat trimap = imread(tri_str, 0);
            resize(trimap, trimap, img.size());

            refineTrimap(trimap);

            clock_t sm_start = clock();
            int sky_mode = is_sky(trimap);
            Mat matte(trimap.size(), trimap.type());
            if(sky_mode == 0){
                cout<<"\t sky_mode : "<<sky_mode;
                Mat mask_0(trimap.size(), trimap.type(), Scalar(0));
                matte = mask_0;
            }else if(sky_mode == 1){
                cout<<"\t sky_mode : "<<sky_mode;
                Mat mask_255(trimap.size(), trimap.type(), Scalar(255));
                matte = mask_255;
            }else{
                SharedMatting sm;
                sm.loadImage(img);
                sm.loadTrimap(trimap);
                sm.solveAlpha();
                matte = sm.save();
            }

            clock_t sm_finish = clock();

            double single_cost_time = (double)(sm_finish - sm_start)/CLOCKS_PER_SEC;
            total_time_cost += single_cost_time;
            cout<<"\tcost time : "<<single_cost_time<<endl;

            Mat concat_result(img.size(), img.type());
            pasteMask(matte, img, concat_result, 0);
            
            string matte_name = filename;
            int index = matte_name.rfind(".");
            matte_name.erase(index, 4);

            imwrite(alpha_path + matte_name + ".png", matte);
            imwrite(concat_path + filename, concat_result);
        }
        cout<<"total_time_cost : \t"<<total_time_cost<<"\t;ave_time_cost :\t"<<total_time_cost/i<<endl;
    }
    return 0;
}
