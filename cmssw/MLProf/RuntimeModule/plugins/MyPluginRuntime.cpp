/*
 * Example plugin to demonstrate the direct multi-threaded inference with TensorFlow 2.
 */

#include <memory>
#include <chrono>
#include <list>
#include <fstream>
#include <random>
#include <stdexcept>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

// define the cache object
// it could handle graph loading and destruction on its own,
// but in this example, we define it as a logicless container
struct CacheData {
  CacheData() : graphDef(nullptr) {}
  std::atomic<tensorflow::GraphDef*> graphDef;
};

class MyPluginRuntime : public edm::stream::EDAnalyzer<edm::GlobalCache<CacheData>> {
public:
  explicit MyPluginRuntime(const edm::ParameterSet&, const CacheData*);
  ~MyPluginRuntime(){};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  // two additional static methods for handling the global cache
  static std::unique_ptr<CacheData> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const CacheData*);

private:
  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

  std::vector<std::string> inputTensorNames_;
  std::vector<std::string> outputTensorNames_;
  std::string filenameOutputCsv_;
  std::string inputType_;
  std::vector<int> inputLengths_;
  std::vector<int> inputSizes_;
  //std:vector batchSize_;
  int nRuns_;
  int nWarmUps_;
  std::vector<int> batchSizes_;

  tensorflow::Session* session_;
};

std::unique_ptr<CacheData> MyPluginRuntime::initializeGlobalCache(const edm::ParameterSet& config) {
  // this method is supposed to create, initialize and return a CacheData instance
  CacheData* cacheData = new CacheData();

  // load the graph def and save it
  std::string graphPath = config.getParameter<std::string>("graphPath");
  cacheData->graphDef = tensorflow::loadGraphDef(graphPath);

  // set tensorflow log leven to warning
  tensorflow::setLogging("2");

  return std::unique_ptr<CacheData>(cacheData);
}

void MyPluginRuntime::globalEndJob(const CacheData* cacheData) {
  // reset the graphDef
  if (cacheData->graphDef != nullptr) {
    delete cacheData->graphDef;
  }
}

void MyPluginRuntime::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // defining this function will lead to a *_cfi file being generated when compiling
  edm::ParameterSetDescription desc;
  desc.add<std::string>("graphPath");
  desc.add<std::vector<std::string>>("inputTensorNames");
  desc.add<std::vector<std::string>>("outputTensorNames");
  desc.add<std::string>("filenameOutputCsv");
  desc.add<std::string>("inputType");
  desc.add<std::vector<int>>("inputLengths");
  desc.add<std::vector<int>>("inputSizes");
  desc.add<std::vector<int>>("batchSizes");
  desc.add<int>("numberRuns");
  desc.add<int>("numberWarmUps");
  descriptions.addWithDefaultLabel(desc);
}

MyPluginRuntime::MyPluginRuntime(const edm::ParameterSet& config, const CacheData* cacheData)
    : inputTensorNames_(config.getParameter<std::vector<std::string>>("inputTensorNames")),
      outputTensorNames_(config.getParameter<std::vector<std::string>>("outputTensorNames")),
      filenameOutputCsv_(config.getParameter<std::string>("filenameOutputCsv")),
      inputType_(config.getParameter<std::string>("inputType")),
      inputLengths_(config.getParameter<std::vector<int>>("inputLengths")),
      inputSizes_(config.getParameter<std::vector<int>>("inputSizes")),
      //batchSize_(config.getParameter<std:vector>("batchSize")),
      nRuns_(config.getParameter<int>("numberRuns")),
      nWarmUps_(config.getParameter<int>("numberWarmUps")),
      batchSizes_(config.getParameter<std::vector<int>>("batchSizes")),
      session_(tensorflow::createSession(cacheData->graphDef)) {}

void MyPluginRuntime::beginJob() {}

void MyPluginRuntime::endJob() {
  // close the session
  tensorflow::closeSession(session_);
}


std::tuple<float,float> mean_and_std(std::vector<float> data_vector)
{
    float runs = data_vector.size();

    // calculate mean runtime
    float mean_runtime = 0;
    for (float time : data_vector){mean_runtime += time;}

    // calculate std of mean runtime
    mean_runtime /= runs;
    float std_runtime = 0;

    for (float time : data_vector){std_runtime += pow(time - mean_runtime, 2);}
    std_runtime /= runs;
    std_runtime = pow(std_runtime, 0.5);

    // wrapp results in tuple
    auto results = std::make_tuple(mean_runtime, std_runtime);
    return results;
}

void deletePreviousFileContent(std::string csv_file)
{
  std::ofstream file(csv_file, std::ios::out | std::ios::trunc);
  file.close();
}

void writeFile(float batch_size, float mean, float std, std::string csv_file)
{
  std::ofstream file(csv_file, std::ios::out| std::ios::app);
  file << batch_size <<","<<mean<<","<<std <<std::endl;
  file.close();
}

void writeFileWholeVector(float batch_size, std::vector<float> runtimes, std::string csv_file)
{
  std::cout << runtimes.size() << std::endl;
  std::ofstream file(csv_file, std::ios::out| std::ios::app);
  for (int i = 0; i < (int) runtimes.size(); i++){
    file << batch_size <<","<< runtimes[i] << std::endl;
  }

  file.close();
}


void print_list(std::list<int> const &list)
{
    for (auto const &i: list) {
        std::cout << i << std::endl;
    }
}

void print_vector(std::vector<int> const &vector)
{
    for (auto const &i: vector) {
        std::cout << i << std::endl;
    }
}

float choose_value(std::string type, int incremental_value, std::default_random_engine generator, std::normal_distribution<float> distribution)
{
  float value;
  if (type == "incremental")
  {
    value = float(incremental_value);
  }
  else if (type == "random")
  {
    value = distribution(generator);
  }
  return value;
}

std::vector<int> get_tensor_shape(tensorflow::Tensor& tensor)
{
    std::vector<int> shape;
    int num_dimensions = tensor.shape().dims();

    for(int ii_dim = 0; ii_dim < num_dimensions; ii_dim++) {
        shape.push_back(tensor.shape().dim_size(ii_dim));
    }
    return shape;
}


void MyPluginRuntime::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  std::list<float> mean_runtimes;
  std::list<float> std_runtimes;
  for (int batchSize : batchSizes_)
  {
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<float> distribution(1.0, 1.0);
    std::vector<tensorflow::Tensor> input_classes_vector;
    int counter = 0;
    tensorflow::Tensor input;
    for (int input_class = 0; input_class < (int)inputLengths_.size(); input_class++)
    {
      // definition and filling of the input tensors for the differents input classes
      // here with if statements -> 1D,2D,3D inputs (creating input tensor with std::vector does not work.)
      if (inputLengths_[input_class]==1)
      {
        input = tensorflow::Tensor(tensorflow::DT_FLOAT, {batchSize, inputSizes_[counter]});
        auto input_eigen_mapped = input.tensor<float, 2>();
        for (int b = 0; b < batchSize; b++)
        {
          for (int i = 0; i < inputSizes_[counter]; i++)
          {
            float value;
            if (inputType_ == "incremental")
            {
              value = float(i);
            }
            else if (inputType_ == "random")
            {
              value = distribution(generator);
            }
            // float value = choose_value(inputType_, i, generator, distribution);
            // input.matrix<float>()(b, i) = value;
            input_eigen_mapped(b, i) = value;
          }
        }
        counter++;
      }
      else if (inputLengths_[input_class]==2)
      {
        input = tensorflow::Tensor(tensorflow::DT_FLOAT, {batchSize, inputSizes_[counter], inputSizes_[counter+1]});
        auto input_eigen_mapped = input.tensor<float, 3>();
        std::cout << input.shape().dims() << std::endl;
        for (int b = 0; b < batchSize; b++)
        {
          for (int i = 0; i < inputSizes_[counter]; i++)
          {
            for (int j = 0; j < inputSizes_[counter+1]; j++)
            {
              // float value = choose_value(inputType_, i+j, generator, distribution);
              float value;
              if (inputType_ == "incremental")
              {
                value = float(i+j);
              }
              else if (inputType_ == "random")
              {
                value = distribution(generator);
              }
              // float value = choose_value(inputType_, i, generator, distribution);
              // input.matrix<float>()(b, i, j) = value;
              input_eigen_mapped(b, i, j) = value;
            }
          }
        }
        counter++;
        counter++;
      }
      else if (inputLengths_[input_class]==3)
      {
        input = tensorflow::Tensor(tensorflow::DT_FLOAT, {batchSize, inputSizes_[counter], inputSizes_[counter+1], inputSizes_[counter+2]});
        auto input_eigen_mapped = input.tensor<float, 4>();
        for (int b = 0; b < batchSize; b++)
        {
          for (int i = 0; i < inputSizes_[counter]; i++)
          {
            for (int j = 0; j < inputSizes_[counter+1]; j++)
            {
              for (int k = 0; k < inputSizes_[counter+2]; k++)
              {
                float value;
                if (inputType_ == "incremental")
                {
                  value = float(i+j+k);
                }
                else if (inputType_ == "random")
                {
                  value = distribution(generator);
                }
                //float value = choose_value(inputType_, i+j+k, generator, distribution);
                // input.matrix<float>()(b, i, j, k) = value;
                input_eigen_mapped(b, i, j, k) = value;
              }
            }
          }
        }
        counter++;
        counter++;
        counter++;
      }
      else
      {
        std::cout << "The inputs must be one-, two- or three dimensional, an error will be thrown" << std::endl;
        throw std::invalid_argument( "input " + std::to_string(input_class) +
                                     " with name " + inputTensorNames_[input_class] +
                                     " has not the right dimension");
      }
      // feeding all input tensors into an input vector


      input_classes_vector.push_back(input);
    }

    // define the output vector
    std::vector<tensorflow::Tensor> outputs;

    // from cmssw source code: inputs in run function is a vector of pairs consisting of a string and a tensor
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    for (int input_class = 0; input_class < (int)inputLengths_.size(); input_class++)
    {
      inputs.push_back({inputTensorNames_[input_class], input_classes_vector[input_class]});
    }


    // run and measure time
    int nRuns = nRuns_; // default 500
    std::vector<float> runtimes;
    int nWarmUps = nWarmUps_; // default 50
    for(int r = 0; r < nRuns + nWarmUps; r++)
    {
      auto start = std::chrono::high_resolution_clock::now();
      // run the graph with given inputs and outputs
      tensorflow::run(session_, inputs, outputTensorNames_, &outputs);

      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> runtime_in_seconds = (end - start);

      if (r >= nWarmUps)
      {
        std::cout << "Current epoch: "<< r - nWarmUps +1 << std::endl;
        // conver runtimes to milli seconds
        runtimes.push_back(runtime_in_seconds.count() * 1000 );
        std::cout << "Corresponding runtime: "<< runtime_in_seconds.count()*1000 << " ms" << std::endl;
      }
      else
      {
        std::cout << "Current warm-up epoch: "<< r + 1 << std::endl;
        continue;
      }
    }
    // calculate metrices
    float mean_runtime=0;
    float std_runtime=0;
    std::tie (mean_runtime, std_runtime) = mean_and_std(runtimes);

    mean_runtimes.push_back(mean_runtime);
    std_runtimes.push_back(std_runtime);

    // save performance not divided by batch size
    //writeFile(batchSize, mean_runtime, std_runtime, filenameOutputCsv_);
    std::cout << "begin writing file" << std::endl;
    auto start_writing = std::chrono::high_resolution_clock::now();
    writeFileWholeVector(batchSize, runtimes, filenameOutputCsv_);
    auto end_writing = std::chrono::high_resolution_clock::now();
    std::cout << "file written" << std::endl;
    std::chrono::duration<float> writing_time = (end_writing - start_writing);
    std::chrono::milliseconds writing_time_ms = std::chrono::duration_cast< std::chrono::milliseconds >( writing_time );
    std::cout << "time taken:" << writing_time.count() << "s" << std::endl;
    std::cout << "time taken:" << writing_time_ms.count() << "ms" << std::endl;

  }
}


DEFINE_FWK_MODULE(MyPluginRuntime);
