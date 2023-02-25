#include "nanodet_openvino.h"

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

template<typename _Tp>
int activation_function_softmax(const _Tp* src, _Tp* dst, int length)
{
    const _Tp alpha = *std::max_element(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}

// 生成中心点
// input_height 高度， input_width 宽度， strides 步长， center_priors 中心点（注意要 x 步长）
static void generate_grid_center_priors(const int input_height, const int input_width, std::vector<int>& strides, std::vector<CenterPrior>& center_priors)
{
    // 遍历不同步长
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int feat_w = ceil((float)input_width / stride);
        int feat_h = ceil((float)input_height / stride);
        for (int y = 0; y < feat_h; y++)
        {
            for (int x = 0; x < feat_w; x++)
            {
                CenterPrior ct;                     // 中心点
                ct.x = x;                           // 1，2，3，连续的，需要 x 步长 ， 才可以得到图片中的真实坐标
                ct.y = y;
                ct.stride = stride;
                center_priors.push_back(ct);
            }
        }
    }
}

// 初始化推理：初始化工作（具体一些内容不确定具体做了什么事情，保留即可）
NanoDet::NanoDet(const char* model_path)
{
    printf("[Jingyu]:begin init model\n");
    // 初始化
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(model_path);     // 模型路径
    
    printf("[Jingyu]:prepare input settings\n");
    // prepare input settings
    InferenceEngine::InputsDataMap inputs_map(model.getInputsInfo());
    input_name_ = inputs_map.begin()->first;
    InferenceEngine::InputInfo::Ptr input_info = inputs_map.begin()->second;
    //input_info->setPrecision(InferenceEngine::Precision::FP32);
    //input_info->setLayout(InferenceEngine::Layout::NCHW);
    
    printf("[Jingyu]:prepare output settings\n");
    //prepare output settings
    InferenceEngine::OutputsDataMap outputs_map(model.getOutputsInfo());
    for (auto &output_info : outputs_map)
    {
        //std::cout<< "Output:" << output_info.first <<std::endl;
        output_info.second->setPrecision(InferenceEngine::Precision::FP32); // 设置了精度，速度不高的时候，或许可以考虑处理一下
    }
    
    //get network
    network_ = ie.LoadNetwork(model, "CPU");                        // 使用CPU推理
    infer_request_ = network_.CreateInferRequest();
    printf("[Jingyu]:end init model\n");
}

NanoDet::~NanoDet()
{
}

// 预处理
void NanoDet::preprocess(cv::Mat& image, InferenceEngine::Blob::Ptr& blob)
{
    int img_w = image.cols;
    int img_h = image.rows;
    int channels = 3;

    InferenceEngine::MemoryBlob::Ptr mblob = InferenceEngine::as<InferenceEngine::MemoryBlob>(blob);
    if (!mblob)
    {
        THROW_IE_EXCEPTION << "We expect blob to be inherited from MemoryBlob in matU8ToBlob, "
            << "but by fact we were not able to cast inputBlob to MemoryBlob";
    }
    // locked memory holder should be alive all time while access to its buffer happens
    auto mblobHolder = mblob->wmap();

    float *blob_data = mblobHolder.as<float *>();


    for (size_t c = 0; c < channels; c++)
    {
        for (size_t  h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                blob_data[c * img_w * img_h + h * img_w + w] =
                    (float)image.at<cv::Vec3b>(h, w)[c];
            }
        }
    }
}

// 检测 input图片， score_threshold得分阈值， nms_threshold后处理阈值
void NanoDet::detect(cv::Mat image, float score_threshold, float nms_threshold, std::vector<BoxInfo>& bbox_dets, std::vector<PtsInfo>& pts_dets)
{
    // printf("[Jingyu]:begin detect\n");
    //auto start = std::chrono::steady_clock::now();

    InferenceEngine::Blob::Ptr input_blob = infer_request_.GetBlob(input_name_);
    
    // 预处理 blob就是一种文件读取
    preprocess(image, input_blob);

    // do inference 推理
    infer_request_.Infer();

    // get output 存放结果
    std::vector<std::vector<BoxInfo>> bbox_results;
    std::vector<std::vector<PtsInfo>> pts_results;  // TODO 新增pts_results推理结果

    bbox_results.resize(this->num_class);           // 二维数组     ->      设置为 num_class 行， 固定了行数
    pts_results.resize(this->num_class);            // TODO 仿照bbox_results生成pts_results
    {
        const InferenceEngine::Blob::Ptr pred_blob = infer_request_.GetBlob(output_name_);

        auto m_pred = InferenceEngine::as<InferenceEngine::MemoryBlob>(pred_blob);
        auto m_pred_holder = m_pred->rmap();
        const float *pred = m_pred_holder.as<const float *>();

        // generate center priors in format of (x, y, stride)
        std::vector<CenterPrior> center_priors;
        generate_grid_center_priors(this->input_size[0], this->input_size[1], this->strides, center_priors);

        // 对推理结果进行解码，pred存放推理的结果
        this->decode_infer(pred, center_priors, score_threshold, bbox_results, pts_results); // TODO 新增pts_result解码
    }

    // 后处理，nms
    // std::vector<BoxInfo> bbox_dets;
    // std::vector<PtsInfo> pts_dets;
    // bbox_results和pts_results大小一样
    for (int i = 0; i < (int)bbox_results.size(); i++)
    {
        this->nms(bbox_results[i], nms_threshold);

        for (auto& box : bbox_results[i])
        {
            bbox_dets.push_back(box);
        }
        for (auto& pts : pts_results[i])            // TODO 更改detect，让其输出pts_results
        {
            pts_dets.push_back(pts);
        }
    }

    //auto end = std::chrono::steady_clock::now();
    //double time = std::chrono::duration<double, std::milli>(end - start).count();
    //std::cout << "inference time:" << time << "ms" << std::endl;
    // return bbox_dets;
}

// 解码推理结果 pred是结果地址， center_priors方便bbox， threshold阈值，
void NanoDet::decode_infer(const float*& pred, std::vector<CenterPrior>& center_priors, float threshold, std::vector<std::vector<BoxInfo>>& bbox_results, std::vector<std::vector<PtsInfo>>& pts_results)
{
    // printf("[Jingyu]:begin decode_infer\n");
    const int num_points = center_priors.size();                // 中心点的个数
    // TODO 增加4个点的部分，num_channels + 8
    const int num_channels = num_class + (reg_max + 1) * 12; //通道的个数
    //printf("num_points:%d\n", num_points);

    //cv::Mat debug_heatmap = cv::Mat::zeros(feature_h, feature_w, CV_8UC3);
    // 遍历所有中心点，解析出所有的内容
    for (int idx = 0; idx < num_points; idx++)
    {
        const int ct_x = center_priors[idx].x;  
        const int ct_y = center_priors[idx].y;
        const int stride = center_priors[idx].stride;

        float score = 0;
        int cur_label = 0;  // 存放得分最高的类别

        // 先获取这个框这里的类别
        for (int label = 0; label < num_class; label++)     // 遍历所有的类别
        {
            if (pred[idx * num_channels + label] / 10 > score)   // 相当于找出类别的得分最大值，它带有的标签即位该类别的标签
            {
                score = pred[idx * num_channels + label];   // float
                cur_label = label;
            }
        }

        if (score > threshold)  // 得分 大于 阈值
        {
            //std::cout << row << "," << col <<" label:" << cur_label << " score:" << score << std::endl;
            const float* bbox_pred = pred + idx * num_channels + num_class;     // 找到bbox的起点

            // 根据dis和中心点解析出bbox并将结果存放到
            bbox_results[cur_label].push_back(this->disPred2Bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride));

            const float* pts_pred = pred + idx * num_channels + num_class + 4 * (reg_max + 1);     // TODO 找到bbox的起点

            pts_results[cur_label].push_back(this->disPred2Pts(pts_pred, cur_label, score, ct_x, ct_y, stride));
            //debug_heatmap.at<cv::Vec3b>(row, col)[0] = 255;
            //cv::imshow("debug", debug_heatmap);
        }

    }
}

// 参考bbox的推理                                                           
PtsInfo NanoDet::pred2Pts(const float*& pred, int label, float score)   // TODO 参考pred2bbox更改成pred2pts
{   
    // 获取各个预测结果
    float x1 = pred[0];
    x1 = (std::max)(x1, .0f);
    x1 = (std::min)(x1, (float)this->input_size[1]);
    float y1 = pred[1];
    y1 = (std::max)(y1, .0f);
    y1 = (std::min)(y1, (float)this->input_size[0]);
    float x2 = pred[2];
    x2 = (std::max)(x2, .0f);
    x2 = (std::min)(x2, (float)this->input_size[1]);
    float y2 = pred[3];     
    y2 = (std::max)(y2, .0f);
    y2 = (std::min)(y2, (float)this->input_size[0]);
    float x3 = pred[4];
    x3 = (std::max)(x3, .0f);
    x3 = (std::min)(x3, (float)this->input_size[1]);
    float y3 = pred[5];
    y3 = (std::max)(y3, .0f);
    y3 = (std::min)(y3, (float)this->input_size[0]);
    float x4 = pred[6]; 
    x4 = (std::max)(x4, .0f);
    x4 = (std::min)(x4, (float)this->input_size[1]);
    float y4 = pred[7];       
    y4 = (std::max)(y4, .0f);
    y4 = (std::min)(y4, (float)this->input_size[0]);
    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    // 装配得到pts信息
    return PtsInfo {x1, y1, x2, y2, x3, y3, x4, y4, score, label};
}

// center point + dis 转换成 bbox
BoxInfo NanoDet::disPred2Bbox(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    // 获取原图的中心点的坐标
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(4); // 存放4个数据
    for (int i = 0; i < 4; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        // 积分得到最终的结果
        for (int j = 0; j < reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    
    // 使用min、max来限制坐标
    // 左上角点
    float xmin = (std::max)(ct_x - dis_pred[0], .0f);   
    float ymin = (std::max)(ct_y - dis_pred[1], .0f);

    // 右下角点
    float xmax = (std::min)(ct_x + dis_pred[2], (float)this->input_size[1]);
    float ymax = (std::min)(ct_y + dis_pred[3], (float)this->input_size[0]);

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    // 得到bbox信息
    return BoxInfo { xmin, ymin, xmax, ymax, score, label };
}


// TODO 新增 center point + dis 转换成 pts
PtsInfo NanoDet::disPred2Pts(const float*& dfl_det, int label, float score, int x, int y, int stride)
{
    // 获取原图的中心点的坐标
    float ct_x = x * stride;
    float ct_y = y * stride;
    std::vector<float> dis_pred;
    dis_pred.resize(8); // 存放8个数据
    for (int i = 0; i < 8; i++)
    {
        float dis = 0;
        float* dis_after_sm = new float[reg_max + 1];
        activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm, reg_max + 1);
        // 积分得到最终的结果
        for (int j = 0; j < reg_max + 1; j++)
        {
            dis += j * dis_after_sm[j];
        }
        dis *= stride;
        //std::cout << "dis:" << dis << std::endl;
        dis_pred[i] = dis;
        delete[] dis_after_sm;
    }
    
    // 使用min、max来限制坐标
    // 左上角点
    float x1 = (std::max)(ct_x - dis_pred[0], .0f);   
    float y1 = (std::max)(ct_y - dis_pred[1], .0f);

    float x2 = (std::max)(ct_x - dis_pred[2], .0f);   
    float y2 = (std::min)(ct_y + dis_pred[3], (float)this->input_size[0]);

    float x3 = (std::min)(ct_x + dis_pred[4], (float)this->input_size[1]);
    float y3 = (std::min)(ct_x + dis_pred[5], (float)this->input_size[0]);

    // 右下角点
    float x4 = (std::min)(ct_x + dis_pred[6], (float)this->input_size[1]);
    float y4 = (std::max)(ct_x - dis_pred[7], .0f); 

    // std::cout << xmin << "," << ymin << "," << xmax << "," << xmax << "," << std::endl;
    // 得到bbox信息
    return PtsInfo { x1, y1, x2, y2, x3, y3, x4, y4, score, label };
}


// nms先不动，还是先以bbox比较iou为主           
// TODO 随后需要更新nms处理策略
void NanoDet::nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}
