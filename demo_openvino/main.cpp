#include "nanodet_openvino.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

struct object_rect {
    int x;
    int y;
    int width;
    int height;
};


// src: 原图
// dst: 目标图
// dst_size: 目标图片的大小
// effect_area: 有效区域
// 大致就是讨论各种情况，然后将原图长宽等比缩放到能放到目标图中(正中间ß)
int resize_uniform(cv::Mat& src, cv::Mat& dst, cv::Size dst_size, object_rect& effect_area)
{
    int w = src.cols;               // 1920
    int h = src.rows;               // 1080
    int dst_w = dst_size.width;     // 320
    int dst_h = dst_size.height;    // 320
    //std::cout << "src: (" << h << ", " << w << ")" << std::endl;
    dst = cv::Mat(cv::Size(dst_w, dst_h), CV_8UC3, cv::Scalar(0));

    float ratio_src = w * 1.0 / h;          // 原图宽高比 w/h < 1
    float ratio_dst = dst_w * 1.0 / dst_h;  // 目标图宽高比 w/h = 1

    int tmp_w = 0;
    int tmp_h = 0;
    if (ratio_src > ratio_dst) {            
        tmp_w = dst_w;                          // tmp_w = 320
        tmp_h = floor((dst_w * 1.0 / w) * h);   // tmp_h = 320 * 1080 / 1920
    }
    else if (ratio_src < ratio_dst) {
        tmp_h = dst_h;
        tmp_w = floor((dst_h * 1.0 / h) * w);
    }
    else {
        cv::resize(src, dst, dst_size);
        effect_area.x = 0;
        effect_area.y = 0;
        effect_area.width = dst_w;
        effect_area.height = dst_h;
        return 0;
    }

    //std::cout << "tmp: (" << tmp_h << ", " << tmp_w << ")" << std::endl;
    cv::Mat tmp;                                
    cv::resize(src, tmp, cv::Size(tmp_w, tmp_h));   // 将原图长宽等比例缩放

    if (tmp_w != dst_w) {                           // 如果宽度没充满 tmp_w = dst_w = 320
        int index_w = floor((dst_w - tmp_w) / 2.0);
        //std::cout << "index_w: " << index_w << std::endl;
        for (int i = 0; i < dst_h; i++) {
            memcpy(dst.data + i * dst_w * 3 + index_w * 3, tmp.data + i * tmp_w * 3, tmp_w * 3);
        }
        effect_area.x = index_w;
        effect_area.y = 0;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else if (tmp_h != dst_h) {                      // 高度没充满, tmp_h < dst_w = 320
        int index_h = floor((dst_h - tmp_h) / 2.0);
        //std::cout << "index_h: " << index_h << std::endl;
        memcpy(dst.data + index_h * dst_w * 3, tmp.data, tmp_w * tmp_h * 3);// 缩放后的图片放到 dst 中间，上下留白
        effect_area.x = 0;                  // effect_area表明了缩放后的原图在dst中的区域 x, y, w, h
        effect_area.y = index_h;
        effect_area.width = tmp_w;
        effect_area.height = tmp_h;
    }
    else {
        printf("error\n");
    }
    //cv::imshow("dst", dst);
    //cv::waitKey(0);
    return 0;
}

// COCO数据集用来给不同类别用不同颜色的框
const int color_list[80][3] =
{
    //{255 ,255 ,255}, //bg
    {216 , 82 , 24},
    {236 ,176 , 31},
    {125 , 46 ,141},
    {118 ,171 , 47},
    { 76 ,189 ,237},
    {238 , 19 , 46},
    { 76 , 76 , 76},
    {153 ,153 ,153},
    {255 ,  0 ,  0},
    {255 ,127 ,  0},
    {190 ,190 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 ,255},
    {170 ,  0 ,255},
    { 84 , 84 ,  0},
    { 84 ,170 ,  0},
    { 84 ,255 ,  0},
    {170 , 84 ,  0},
    {170 ,170 ,  0},
    {170 ,255 ,  0},
    {255 , 84 ,  0},
    {255 ,170 ,  0},
    {255 ,255 ,  0},
    {  0 , 84 ,127},
    {  0 ,170 ,127},
    {  0 ,255 ,127},
    { 84 ,  0 ,127},
    { 84 , 84 ,127},
    { 84 ,170 ,127},
    { 84 ,255 ,127},
    {170 ,  0 ,127},
    {170 , 84 ,127},
    {170 ,170 ,127},
    {170 ,255 ,127},
    {255 ,  0 ,127},
    {255 , 84 ,127},
    {255 ,170 ,127},
    {255 ,255 ,127},
    {  0 , 84 ,255},
    {  0 ,170 ,255},
    {  0 ,255 ,255},
    { 84 ,  0 ,255},
    { 84 , 84 ,255},
    { 84 ,170 ,255},
    { 84 ,255 ,255},
    {170 ,  0 ,255},
    {170 , 84 ,255},
    {170 ,170 ,255},
    {170 ,255 ,255},
    {255 ,  0 ,255},
    {255 , 84 ,255},
    {255 ,170 ,255},
    { 42 ,  0 ,  0},
    { 84 ,  0 ,  0},
    {127 ,  0 ,  0},
    {170 ,  0 ,  0},
    {212 ,  0 ,  0},
    {255 ,  0 ,  0},
    {  0 , 42 ,  0},
    {  0 , 84 ,  0},
    {  0 ,127 ,  0},
    {  0 ,170 ,  0},
    {  0 ,212 ,  0},
    {  0 ,255 ,  0},
    {  0 ,  0 , 42},
    {  0 ,  0 , 84},
    {  0 ,  0 ,127},
    {  0 ,  0 ,170},
    {  0 ,  0 ,212},
    {  0 ,  0 ,255},
    {  0 ,  0 ,  0},
    { 36 , 36 , 36},
    { 72 , 72 , 72},
    {109 ,109 ,109},
    {145 ,145 ,145},
    {182 ,182 ,182},
    {218 ,218 ,218},
    {  0 ,113 ,188},
    { 80 ,182 ,188},
    {127 ,127 ,  0},
};

// 原图，推理得到的bbox，有效区域effect_roiß
void draw_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes, const std::vector<PtsInfo>& points, object_rect effect_roi)
{
    // 类别名称
    static const char* class_names[] = {
                                        "B_G", "B_1", "B_2", "B_3", "B_4", "B_5", "B_O", "B_Bs", "B_Bb",        // TODO: 更改绘制时，给打的标签
                                        "R_G", "R_1", "R_2", "R_3", "R_4", "R_5", "R_O", "R_Bs", "R_Bb",
                                        "N_G", "N_1", "N_2", "N_3", "N_4", "N_5", "N_O", "N_Bs", "N_Bb",
                                        "P_G", "P_1", "P_2", "P_3", "P_4", "P_5", "P_O", "P_Bs", "P_Bb"
    };

    // 拷贝一份图片
    cv::Mat image = bgr.clone();
    // 原图的尺寸大小
    int src_w = image.cols;
    int src_h = image.rows;
    // 目标的尺寸大小
    int dst_w = effect_roi.width;
    int dst_h = effect_roi.height;
    // 宽度比，高度比
    float width_ratio = (float)src_w / (float)dst_w;    // 1920 / 320
    float height_ratio = (float)src_h / (float)dst_h;   // 1080 / (320 * 1080 / 1920)

    // 遍历所有bbox区域
    for (size_t i = 0; i < bboxes.size(); i++)
    {
        // 获取bbox信息
        const BoxInfo& bbox = bboxes[i];
        const PtsInfo& pts = points[i];
        
        // 根据预测类别label标签，获取该类别对应的一种颜色
        cv::Scalar color = cv::Scalar(color_list[bbox.label][0], color_list[bbox.label][1], color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        // 在原图画出来矩形 bbox坐标是320x320图中的坐标ß
        // (bbox.x1 - effect_roi.x) * width_ratio 得到原图中的横坐标， 其中bbox.x1 - effect_roi.x 得到的是相对感兴趣区域的横坐标
        // (bbox.y1 - effect_roi.y) * height_ratio 得到原图中的纵坐标  其中bbox.x1 - effect_roi.x 得到的是相对感兴趣区域的纵坐标
        cv::rectangle(image, cv::Rect(cv::Point((bbox.x1 - effect_roi.x) * width_ratio, (bbox.y1 - effect_roi.y) * height_ratio),
                                      cv::Point((bbox.x2 - effect_roi.x) * width_ratio, (bbox.y2 - effect_roi.y) * height_ratio)), color);

        // TODO 新增交点的绘制
        cv::line(image, cv::Point(pts.x1*width_ratio, pts.y1*height_ratio), cv::Point(pts.x2*width_ratio, pts.y2*height_ratio), color);
        cv::line(image, cv::Point(pts.x2*width_ratio, pts.y2*height_ratio), cv::Point(pts.x3*width_ratio, pts.y3*height_ratio), color);
        cv::line(image, cv::Point(pts.x3*width_ratio, pts.y3*height_ratio), cv::Point(pts.x4*width_ratio, pts.y4*height_ratio), color);
        cv::line(image, cv::Point(pts.x4*width_ratio, pts.y4*height_ratio), cv::Point(pts.x1*width_ratio, pts.y1*height_ratio), color);
        
        
        // 文本，标出类别提及得分
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        // 在bbox的左上角标明文本
        int x = (bbox.x1 - effect_roi.x) * width_ratio;
        int y = (bbox.y1 - effect_roi.y) * height_ratio - label_size.height - baseLine;
        
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        // 标出文本框
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            color, -1);
        
        // 在图片上标注ß文本
        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
}

// 处理图片
int image_demo(NanoDet& detector, const char* imagepath)
{
    printf("[Jingyu]:begin inference\n");
    // const char* imagepath = "D:/Dataset/coco/val2017/*.jpg";
    std::vector<std::string> filenames;     
    cv::glob(imagepath, filenames, false);    
    int height = detector.input_size[0];
    int width = detector.input_size[1];

    for (auto img_name : filenames)             // 处理文件夹中的每一张图片
    {
        cv::Mat image = cv::imread(img_name);   // 读取图片
        if (image.empty())
        {
            fprintf(stderr, "cv::imread %s failed\n", img_name);
            return -1;
        }
        object_rect effect_roi;                     // roi区域
        cv::Mat resized_img;                        // 图片大小resize
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);
        // 生成结果
        std::vector<BoxInfo> bbox_dets;
        std::vector<PtsInfo> pts_dets;
        detector.detect(resized_img, 0.4, 0.5, bbox_dets, pts_dets);
        // 绘制出结果
        draw_bboxes(image, bbox_dets, pts_dets, effect_roi);
        cv::waitKey(0);

    }
    return 0;
}

// 根据相机推理
int webcam_demo(NanoDet& detector, int cam_id)
{
    cv::Mat image;
    cv::VideoCapture cap(cam_id);   //  获取相机

    int height = detector.input_size[0];    // height
    int width = detector.input_size[1];     // width

    while (true)
    {
        cap >> image;               // 读取图片
        object_rect effect_roi;     // 包含x,y,w,h的一个结构   
        cv::Mat resized_img;        // resize处理后的图片
        // 原图，目标大小图，目标图尺寸大小，原图缩放后在resized_img真实的区域
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);
        // 对原图进行推理
        std::vector<BoxInfo> bbox_dets;
        std::vector<PtsInfo> pts_dets;
        detector.detect(resized_img, 0.4, 0.5, bbox_dets, pts_dets);

        // 在原图上绘制出结果， effect_roi为原图所在真实区域
        draw_bboxes(image, bbox_dets, pts_dets, effect_roi);
        cv::waitKey(1);
    }
    return 0;
}

int video_demo(NanoDet& detector, const char* path)
{
    cv::Mat image;
    cv::VideoCapture cap(path);

    int height = detector.input_size[0];
    int width = detector.input_size[1];

    while (true)
    {
        cap >> image;
        object_rect effect_roi;
        cv::Mat resized_img;
        resize_uniform(image, resized_img, cv::Size(width, height), effect_roi);
        std::vector<BoxInfo> bbox_dets;
        std::vector<PtsInfo> pts_dets;
        detector.detect(resized_img, 0.2, 0.2, bbox_dets, pts_dets);

        // 在原图上绘制出结果， effect_roi为原图所在真实区域
        draw_bboxes(image, bbox_dets, pts_dets, effect_roi);
        cv::waitKey(1);
    }
    return 0;
}

int benchmark(NanoDet& detector)
{
    int loop_num = 100;
    int warm_up = 8;

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    int height = detector.input_size[0];
    int width = detector.input_size[1];
    cv::Mat image(height, width, CV_8UC3, cv::Scalar(1, 1, 1));

    for (int i = 0; i < warm_up + loop_num; i++)
    {
        auto start = std::chrono::steady_clock::now();
        std::vector<BoxInfo> bbox_dets;
        std::vector<PtsInfo> pts_dets;
        detector.detect(image, 0.4, 0.5, bbox_dets, pts_dets);
        auto end = std::chrono::steady_clock::now();
        double time = std::chrono::duration<double, std::milli>(end - start).count();
        if (i >= warm_up)
        {
            time_min = (std::min)(time_min, time);
            time_max = (std::max)(time_max, time);
            time_avg += time;
        }
    }
    time_avg /= loop_num;
    fprintf(stderr, "%20s  min = %7.2f  max = %7.2f  avg = %7.2f\n", "nanodet", time_min, time_max, time_avg);
    return 0;
}


int main(int argc, char** argv)
{
    // if (argc != 3)      // 首先检测参数，确定参数的内容是否符合要求
    // {
    //     fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
    //     return -1;
    // }
    std::cout<<"start init model"<<std::endl;

    // model path
    // auto detector = NanoDet("nanodet.xml"); // TODO 根据模型路径初始化检测器，需要更换存放路径 
    auto detector = NanoDet("/home/jingyu/nanodet.xml");
    std::cout<<"success"<<std::endl;

    int mode = 2;               // 推理模式的选择 相机读取，图片推理，视频推理
    switch (mode)
    {
    case 0:{
        // 相机模式
        int cam_id = atoi(argv[2]);
        webcam_demo(detector, cam_id);
        break;
        }
    case 1:{
        // 图片模式
        const char* images = "/home/jingyu/pic/";
        image_demo(detector, images);
        break;
        }
    case 2:{
        // 视频模式
        const char* path = "/home/jingyu/test.mp4";
        video_demo(detector, path);
        break;
        }
    case 3:{
        benchmark(detector);
        break;
        }
    default:{
        fprintf(stderr, "usage: %s [mode] [path]. \n For webcam mode=0, path is cam id; \n For image demo, mode=1, path=xxx/xxx/*.jpg; \n For video, mode=2; \n For benchmark, mode=3 path=0.\n", argv[0]);
        break;
        }
    }
}
