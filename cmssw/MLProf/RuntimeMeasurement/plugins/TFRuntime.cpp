/*
 * Plugin to measure the runtime of a tensorflow graph.
 */

#include <chrono>
#include <fstream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

#include "MLProf/Utils/interface/utils.h"

class TFRuntime : public edm::stream::EDAnalyzer<edm::GlobalCache<tensorflow::SessionCache>> {
public:
  explicit TFRuntime(const edm::ParameterSet&, const tensorflow::SessionCache*);
  ~TFRuntime(){};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<tensorflow::SessionCache> initializeGlobalCache(const edm::ParameterSet&);
  static void globalEndJob(const tensorflow::SessionCache*);

private:
  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

  inline float drawNormal() { return normalPdf_(rndGen_); }
  tensorflow::Tensor createInputTensor(int rank, std::vector<int> shape);

  // parameters
  std::vector<std::string> inputTensorNames_;
  std::vector<std::string> outputTensorNames_;
  std::string outputFile_;
  std::string inputTypeStr_;
  std::vector<int> inputRanks_;
  std::vector<int> flatInputSizes_;
  std::vector<int> batchSizes_;
  int nCalls_;

  // other members
  int nInputs_;
  int nPreCalls_;
  mlprof::InputType inputType_;
  std::random_device rnd_;
  std::default_random_engine rndGen_;
  std::normal_distribution<float> normalPdf_;
  const tensorflow::Session* session_;
};

std::unique_ptr<tensorflow::SessionCache> TFRuntime::initializeGlobalCache(const edm::ParameterSet& params) {
  std::string graphPath = edm::FileInPath(params.getParameter<std::string>("graphPath")).fullPath();
  // cpu-only for now
  tensorflow::Options options{tensorflow::Backend::cpu};
  return std::make_unique<tensorflow::SessionCache>(graphPath, options);
}

void TFRuntime::globalEndJob(const tensorflow::SessionCache* cache) {}

void TFRuntime::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // the path to the file containing the graph
  desc.add<std::string>("graphPath");
  // the names of the input tensors
  desc.add<std::vector<std::string>>("inputTensorNames");
  // the names of the output tensors
  desc.add<std::vector<std::string>>("outputTensorNames");
  // the name of the output csv file
  desc.add<std::string>("outputFile");
  // the type of input values, either "incremental" or "random"
  desc.add<std::string>("inputType", "random");
  // the rank (number of dimensions) of each input tensor
  desc.add<std::vector<int>>("inputRanks");
  // flat list of sizes of each dimension of each input tensor
  // (for a graph with a 1D and a 2D input tensor, this would be a vector of three values)
  desc.add<std::vector<int>>("flatInputSizes");
  // batch sizes to test
  desc.add<std::vector<int>>("batchSizes");
  // the number of calls to the graph to measure the runtime
  desc.add<int>("nCalls");

  descriptions.addWithDefaultLabel(desc);
}

TFRuntime::TFRuntime(const edm::ParameterSet& config, const tensorflow::SessionCache* cache)
    : inputTensorNames_(config.getParameter<std::vector<std::string>>("inputTensorNames")),
      outputTensorNames_(config.getParameter<std::vector<std::string>>("outputTensorNames")),
      outputFile_(config.getParameter<std::string>("outputFile")),
      inputTypeStr_(config.getParameter<std::string>("inputType")),
      inputRanks_(config.getParameter<std::vector<int>>("inputRanks")),
      flatInputSizes_(config.getParameter<std::vector<int>>("flatInputSizes")),
      batchSizes_(config.getParameter<std::vector<int>>("batchSizes")),
      nCalls_(config.getParameter<int>("nCalls")),
      nInputs_(inputTensorNames_.size()),
      nPreCalls_(10),
      rndGen_(rnd_()),
      normalPdf_(0.0, 1.0),
      session_(cache->getSession()) {
  // the number of input ranks must match the number of input tensors
  if ((int)inputRanks_.size() != nInputs_) {
    throw cms::Exception("InvalidInputRanks") << "number of input ranks must match number of input tensors";
  }
  // the input must be at least 1 dimensional
  for (auto rank : inputRanks_) {
    if (rank < 1) {
      throw cms::Exception("InvalidRank") << "only ranks above 0 are supported, got " << rank;
    }
  }
  // the sum of ranks must match the number of flat input sizes
  if (std::accumulate(inputRanks_.begin(), inputRanks_.end(), 0) != (int)flatInputSizes_.size()) {
    throw cms::Exception("InvalidFlatInputSizes")
        << "sum of input ranks must match number of flat input sizes, got " << flatInputSizes_.size();
  }
  // batch size must be positive
  for (auto batchSize : batchSizes_) {
    if (batchSize < 1) {
      throw cms::Exception("InvalidBatchSize") << "batch sizes must be positive, got " << batchSize;
    }
  }
  // input sizes must be positive
  for (auto size : flatInputSizes_) {
    if (size < 1) {
      throw cms::Exception("InvalidInputSize") << "input sizes must be positive, got " << size;
    }
  }
  // check the input type
  if (inputTypeStr_ == "incremental") {
    inputType_ = mlprof::InputType::Incremental;
  } else if (inputTypeStr_ == "random") {
    inputType_ = mlprof::InputType::Random;
  } else if (inputTypeStr_ == "zeros") {
    inputType_ = mlprof::InputType::Zeros;
  } else {
    throw cms::Exception("InvalidInputType")
        << "input type must be either 'incremental', 'zeros' or 'random', got " << inputTypeStr_;
  }
}

void TFRuntime::beginJob() {}

void TFRuntime::endJob() {}

tensorflow::Tensor TFRuntime::createInputTensor(int rank, std::vector<int> shape) {
  // convert the shape to a tf shape
  tensorflow::TensorShape tShape;
  for (auto dim : shape) {
    tShape.AddDim(dim);
  }

  // create the tensor
  tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tShape);

  // fill it
  float* data = tensor.flat<float>().data();
  for (int i = 0; i < tensor.NumElements(); i++, data++) {
    *data = inputType_ == mlprof::InputType::Incremental ? float(i) :
    inputType_ == mlprof::InputType::Zeros ? float(0) :
    drawNormal();
  }

  return tensor;
}

void TFRuntime::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  for (int batchSize : batchSizes_) {
    // prepare inputs
    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs;
    int sizeOffset = 0;
    for (int i = 0; i < nInputs_; i++) {
      // build the shape
      std::vector<int> shape = {batchSize};
      for (int j = 0; j < inputRanks_[i]; j++, sizeOffset++) {
        shape.push_back(flatInputSizes_[sizeOffset]);
      }
      // create and save it
      inputs.push_back({inputTensorNames_[i], createInputTensor(inputRanks_[i], shape)});
    }

    // prepare output vectors
    std::vector<tensorflow::Tensor> outputs;

    // pre calls to "warm up"
    for (int r = 0; r < nPreCalls_; r++) {
      tensorflow::run(session_, inputs, outputTensorNames_, &outputs);
    }

    // actual calls to measure runtimes
    std::vector<float> runtimes;
    for (int r = 0; r < nCalls_; r++) {
      auto start = std::chrono::high_resolution_clock::now();
      tensorflow::run(session_, inputs, outputTensorNames_, &outputs);
      auto end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<float> runtime_in_seconds = (end - start);
      runtimes.push_back(runtime_in_seconds.count() * 1000);
    }

    // save them
    mlprof::writeRuntimes(outputFile_, batchSize, runtimes);
  }
}

DEFINE_FWK_MODULE(TFRuntime);
