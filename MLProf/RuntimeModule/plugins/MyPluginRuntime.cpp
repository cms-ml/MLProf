/*
 * Example plugin to demonstrate the direct multi-threaded inference with TensorFlow 2.
 */

#include <memory>
#include <chrono>
#include <list>
#include <fstream>

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

  std::string inputTensorName_;
  std::string outputTensorName_;
  std::string filenameOutputCsv_;
  int inputSize_;
  int outputSize_;
  //std:vector batchSize_;
  int nRuns_;
  int nWarmUps_;
  std::vector<int> batchsizes_;

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
  desc.add<std::string>("inputTensorName");
  desc.add<std::string>("outputTensorName");
  desc.add<std::string>("filenameOutputCsv");
  desc.add<int>("inputSize");
  desc.add<int>("outputSize");
  desc.add<std::vector<int>>("batchsizes");
  desc.add<int>("numberRuns");
  desc.add<int>("numberWarmUps");
  descriptions.addWithDefaultLabel(desc);
}

MyPluginRuntime::MyPluginRuntime(const edm::ParameterSet& config, const CacheData* cacheData)
    : inputTensorName_(config.getParameter<std::string>("inputTensorName")),
      outputTensorName_(config.getParameter<std::string>("outputTensorName")),
      filenameOutputCsv_(config.getParameter<std::string>("filenameOutputCsv")),
      inputSize_(config.getParameter<int>("inputSize")),
      outputSize_(config.getParameter<int>("outputSize")),
      //batchSize_(config.getParameter<std:vector>("batchSize")),
      nRuns_(config.getParameter<int>("numberRuns")),
      nWarmUps_(config.getParameter<int>("numberWarmUps")),
      batchsizes_(config.getParameter<std::vector<int>>("batchsizes")),
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


void MyPluginRuntime::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  std::vector<int> batchsizes=batchsizes_;
  print_vector(batchsizes_);
  print_vector(batchsizes);
  std::list<float> mean_runtimes;
  std::list<float> std_runtimes;
  for (int batchsize : batchsizes)
  {
    tensorflow::Tensor input(tensorflow::DT_FLOAT, {batchsize, inputSize_});
    for (size_t i = 0; i < 10; i++)
    {
      input.matrix<float>()(0, i) = float(i);
    }

    // define the output
    std::vector<tensorflow::Tensor> outputs;

    // run and measure time
    int nRuns= nRuns_; //500
    std::vector<float> runtimes;
    int warm_up= nWarmUps_; //100
    for(int r = 0; r < nRuns + warm_up; r++)
    {
      auto start = std::chrono::high_resolution_clock::now();
      tensorflow::run(session_, {{inputTensorName_, input}}, {outputTensorName_}, &outputs);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> runtime_in_seconds = (end - start);

      if (r > warm_up)
      {
        std::cout << "Current epoch: "<< r - warm_up << std::endl;
        // conver runtimes to milli seconds
        runtimes.push_back(runtime_in_seconds.count() * 1000 );
        std::cout << "Corresponding runtime: "<< runtime_in_seconds.count()*1000 << " ms" << std::endl;
      }
      else
      {
        std::cout << "Current warm-up epoch: "<< r << std::endl;
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
    writeFile(batchsize, mean_runtime, std_runtime, filenameOutputCsv_);
  }
}


DEFINE_FWK_MODULE(MyPluginRuntime);
