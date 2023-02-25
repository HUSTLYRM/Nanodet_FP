//
// Create by RangiLyu
// 2021 / 1 / 12
//

#ifndef _NANODET_OPENVINO_H_
#define _NANODET_OPENVINO_H_

#include <string>
#include <opencv2/core.hpp>
#include <inference_engine.hpp>


// 检测头信息
typedef struct HeadInfo
{
    std::string cls_layer;
    std::string dis_layer;
    int stride;
} HeadInfo;

// 中心点prior
struct CenterPrior
{
    int x;
    int y;
    int stride;
};

// 检验框信息
typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

// 四点信息
typedef struct PtsInfo                                  // TODO： 新增PtsInfo四点信息
{
    float x1;
    float y1;
    float x2;
    float y2;
    float x3;
    float y3;
    float x4;
    float y4;
    float score;
    int label;
} PtsInfo;

// 检测类
class NanoDet
{
public:
    NanoDet(const char* param);

    ~NanoDet();

    InferenceEngine::ExecutableNetwork network_;

    InferenceEngine::InferRequest infer_request_;
    // static bool hasGPU;

    // modify these parameters to the same with your config if you want to use your own model
    // TODO: input_size修改成和模型大小一样
    int input_size[2] = {416, 416}; // input height and width   
    // TODO: num_class修改成和模型一致
    int num_class = 36; // number of classes. 80 for COCO   
    // TODO: reg_max和模型一致就可以
    int reg_max = 7; // `reg_max` set in the training config. Default: 7.  
    // TODO: strides和模型一致
    std::vector<int> strides = { 8, 16, 32, 64 }; // strides of the multi-level feature.   

    // 关键函数: detect，分数阈值，nms阈值
    void detect(cv::Mat image, float score_threshold, float nms_threshold, std::vector<BoxInfo>& bbox_dets, std::vector<PtsInfo>& pts_dets);

private:

    void preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob);
    
    void decode_infer(const float*& pred, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& bbox_results, std::vector<std::vector<PtsInfo>>& pts_results);
    
    BoxInfo disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride);
    PtsInfo disPred2Pts(const float*& dfl_det, int label, float score, int x, int y, int stride);
    PtsInfo pred2Pts(const float*& pred, int label, float score);

    static void nms(std::vector<BoxInfo>& result, float nms_threshold);
    
    std::string input_name_ = "data";
    std::string output_name_ = "output";
};


#endif //_NANODE_TOPENVINO_H_
