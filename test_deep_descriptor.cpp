#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <pcl/point_cloud.h>
#include "SubmapContext/nanoflann.hpp"
#include "SubmapContext/KDTreeVectorOfVectorsAdaptor.h"


using namespace std;
using KeyMat = std::vector<std::vector<float> >;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor< KeyMat, float >;

std::vector<float> read_radar_data(const std::string radar_data_path)
{
    std::cout << "try open file:" << radar_data_path << std::endl;
    std::vector<double> radar_data_buffer_double;
    std::ifstream radar_data_file(radar_data_path, std::ifstream::in | std::ifstream::binary);
    if (radar_data_file.is_open())
    {
        radar_data_file.seekg(0, std::ios::end);
        size_t num_elements = radar_data_file.tellg() / sizeof(double);
        radar_data_file.seekg(0, std::ios::beg);
        radar_data_buffer_double.resize(num_elements);
        radar_data_file.read(reinterpret_cast<char*>(&radar_data_buffer_double[0]), num_elements*sizeof(double));
        radar_data_file.close();
    }
    else
    {
        std::cout << "Unable to open file:" << radar_data_path << std::endl;
    }

    std::vector<float> radar_data_buffer(radar_data_buffer_double.begin(), radar_data_buffer_double.end());
    // std::cout << "radar_data_buffer:" <<radar_data_buffer <<std::endl;
    return radar_data_buffer;
}


std::vector<float> read_radar_database(const std::string radar_data_path)
{
    std::cout << "try open file:" << radar_data_path << std::endl;
    std::vector<float> radar_data_buffer_float;
    std::ifstream radar_data_file(radar_data_path, std::ifstream::in | std::ifstream::binary);
    if (radar_data_file.is_open())
    {
        radar_data_file.seekg(0, std::ios::end);
        size_t num_elements = radar_data_file.tellg() / sizeof(float);
        radar_data_file.seekg(0, std::ios::beg);
        radar_data_buffer_float.resize(num_elements);
        radar_data_file.read(reinterpret_cast<char*>(&radar_data_buffer_float[0]), num_elements*sizeof(float));
        radar_data_file.close();
    }
    else
    {
        std::cout << "Unable to open file:" << radar_data_path << std::endl;
    }

    std::vector<float> radar_data_buffer(radar_data_buffer_float.begin(), radar_data_buffer_float.end());
    return radar_data_buffer;
}


int main()
{
    std::string database_path, radar_data_path, model_path;
    database_path = "./data_deep/seq1_database.bin";
    radar_data_path = "./data_deep/000000.bin";
    model_path ="./data_deep//PointNetVlad.pt";
    int num_points = 1280;
    int input_dim = 4; 
    int output_dim = 256;

    // load databases
    std::vector<float> database_data = read_radar_database(database_path);
    std::cout << "Database data size: " << database_data.size() << std::endl;
    int n = database_data.size() / output_dim;
    KeyMat database_des(n, std::vector<float>(output_dim));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < output_dim; ++j) {
            database_des[i][j] = database_data[i * output_dim + j];
        }
    }

    // load queries
    std::vector<float> radar_data = read_radar_data(radar_data_path);
    float input[num_points*input_dim];
    for (int i = 0; i < radar_data.size(); i++)
    {
        input[i] = radar_data[i];
    }

    torch::Tensor tester = torch::from_blob(input, {1, 1, num_points, input_dim}, torch::kFloat); // to tensor
    std::cout << "tester tensor shape: " << tester.sizes() << std::endl;
    std::cout << "tester tensor values: " << tester << std::endl;

    // load model
    torch::jit::script::Module module = torch::jit::load(model_path);
    module.eval();
    std::cout << "model loaded" << std::endl;

    // gen query descriptor
    torch::Tensor query_des = module.forward({tester}).toTensor();
    std::cout << "query_des:" << query_des <<std::endl;

    // Convert tensor to std::vector<float>
    query_des = query_des.view({-1});
    std::vector<float> query_des_vector(query_des.data_ptr<float>(), query_des.data_ptr<float>() + query_des.numel());
    
    // place recognition
    int NUM_CANDIDATES_FROM_TREE = 20;
    KeyMat des_to_search_;
    des_to_search_.clear();
    des_to_search_.assign(database_des.begin(), database_des.end()) ;

    std::unique_ptr<InvKeyTree> database_tree_;
    database_tree_.reset(); 
    database_tree_ = std::unique_ptr<InvKeyTree>(new InvKeyTree(output_dim /* dim */, des_to_search_, 10 /* max leaf */ ));

     // knn search
    std::vector<size_t> candidate_indexes( NUM_CANDIDATES_FROM_TREE ); 
    std::vector<float> out_dists_sqr( NUM_CANDIDATES_FROM_TREE );

    // TicTocV2 t_tree_search;
    nanoflann::KNNResultSet<float> knnsearch_result( NUM_CANDIDATES_FROM_TREE );
    knnsearch_result.init( &candidate_indexes[0], &out_dists_sqr[0] );

    std::cout << "query_des_vector:" << query_des_vector <<std::endl;
    database_tree_->index->findNeighbors( knnsearch_result, &query_des_vector[0] /* query */, nanoflann::SearchParams(10)); 
    std::cout << "candidate_indexes:" << candidate_indexes << std::endl;
    std::cout << "out_dists_sqr:" << out_dists_sqr << std::endl;

}
