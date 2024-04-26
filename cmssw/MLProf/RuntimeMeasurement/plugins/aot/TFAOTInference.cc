/*
 * Plugin to measure the inference runtime of an ahead-of-time (AOT) compiled tensorflow model.
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
#include "PhysicsTools/TensorFlowAOT/interface/Model.h"

#include "MLProf/Utils/interface/utils.h"
#include "tfaot-model-mlprof-test/model.h"

class TFAOTInference : public edm::stream::EDAnalyzer<> {
public:
  explicit TFAOTInference(const edm::ParameterSet&);
  ~TFAOTInference(){};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void beginJob(){};
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob(){};

  tfaot::BoolArrays createBoolInput(size_t shape1);
  tfaot::Int32Arrays createInt32Input(size_t shape1);
  tfaot::Int64Arrays createInt64Input(size_t shape1);
  tfaot::FloatArrays createFloatInput(size_t shape1);
  tfaot::DoubleArrays createDoubleInput(size_t shape1);

  inline bool drawBool() { return bernoulliPDF_(rndGen_); }
  inline float drawNormalFloat() { return normalPDFFloat_(rndGen_); }
  inline double drawNormalDouble() { return normalPDFDouble_(rndGen_); }

  // parameters
  std::string outputFile_;
  std::string inputTypeStr_;
  std::vector<std::string> batchRules_;
  int batchSize_;
  int nCalls_;

  // other members
  int nPreCalls_;
  mlprof::InputType inputType_;
  std::random_device rnd_;
  std::default_random_engine rndGen_;
  std::bernoulli_distribution bernoulliPDF_;
  std::normal_distribution<float> normalPDFFloat_;
  std::normal_distribution<double> normalPDFDouble_;

  // aot model
  // INSERT=model
};

void TFAOTInference::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // the name of the output csv file
  desc.add<std::string>("outputFile");
  // the type of input values, either "incremental" or "random"
  desc.add<std::string>("inputType", "random");
  // list of batch rules in the format "5:1,4" or "5:2,2,2" (padding is inferred)
  desc.add<std::vector<std::string>>("batchRules", std::vector<std::string>());
  // batch size to test
  desc.add<int>("batchSize");
  // the number of calls to the graph to measure the runtime
  desc.add<int>("nCalls");

  descriptions.addWithDefaultLabel(desc);
}

TFAOTInference::TFAOTInference(const edm::ParameterSet& config)
    : outputFile_(config.getParameter<std::string>("outputFile")),
      inputTypeStr_(config.getParameter<std::string>("inputType")),
      batchRules_(config.getParameter<std::vector<std::string>>("batchRules")),
      batchSize_(config.getParameter<int>("batchSize")),
      nCalls_(config.getParameter<int>("nCalls")),
      nPreCalls_(10),
      rndGen_(rnd_()),
      bernoulliPDF_(0.5),
      normalPDFFloat_(0.0, 1.0),
      normalPDFDouble_(0.0, 1.0) {
  // batch size must be positive
  if (batchSize_ < 1) {
    throw cms::Exception("InvalidBatchSize") << "batch size must be positive, got " << batchSize_;
  }

  // check the input type
  if (inputTypeStr_ == "incremental") {
    inputType_ = mlprof::InputType::Incremental;
  } else if (inputTypeStr_ == "random") {
    inputType_ = mlprof::InputType::Random;
  } else if (inputTypeStr_ == "zeros") {
    inputType_ = mlprof::InputType::Zeros;
} else if (inputTypeStr_ == "ones") {
    inputType_ = mlprof::InputType::Ones;
  } else {
    throw cms::Exception("InvalidInputType") << "input type unknown: " << inputTypeStr_;
  }

  // register batch rules
  for (const auto& rule : batchRules_) {
    model_.setBatchRule(rule);
    std::cout << "registered batch rule " << rule << std::endl;
  }
}

tfaot::BoolArrays TFAOTInference::createBoolInput(size_t shape1) {
  tfaot::BoolArrays input;
  for (int i = 0; i < batchSize_; i++) {
    std::vector<bool> vec(shape1);
    for (size_t j = 0; j < shape1; j++) {
      switch (inputType_) {
        case mlprof::InputType::Zeros:
          vec[i] = false;
          break;
      case mlprof::InputType::Ones:
          vec[i] = true;
          break;
        case mlprof::InputType::Random:
        case mlprof::InputType::Incremental: // not implemented
          vec[i] = drawBool();
          break;
      }
    }
    input.push_back(vec);
  }
  return input;
}

tfaot::Int32Arrays TFAOTInference::createInt32Input(size_t shape1) {
  tfaot::Int32Arrays input;
  for (int i = 0; i < batchSize_; i++) {
    std::vector<int32_t> vec(shape1);
    for (size_t j = 0; j < shape1; j++) {
      switch (inputType_) {
        case mlprof::InputType::Zeros:
        case mlprof::InputType::Random:  // not implemented
          vec[i] = 0;
          break;
        case mlprof::InputType::Ones:
          vec[i] = 1;
          break;
        case mlprof::InputType::Incremental:
          vec[i] = (int32_t)j;
          break;
      }
    }
    input.push_back(vec);
  }
  return input;
}

tfaot::Int64Arrays TFAOTInference::createInt64Input(size_t shape1) {
  tfaot::Int64Arrays input;
  for (int i = 0; i < batchSize_; i++) {
    std::vector<int64_t> vec(shape1);
    for (size_t j = 0; j < shape1; j++) {
      switch (inputType_) {
        case mlprof::InputType::Zeros:
        case mlprof::InputType::Random:  // not implemented
          vec[i] = 0;
          break;
        case mlprof::InputType::Ones:
          vec[i] = 1;
          break;
        case mlprof::InputType::Incremental:
          vec[i] = (int64_t)j;
          break;
      }
    }
    input.push_back(vec);
  }
  return input;
}

tfaot::FloatArrays TFAOTInference::createFloatInput(size_t shape1) {
  tfaot::FloatArrays input;
  for (int i = 0; i < batchSize_; i++) {
    std::vector<float> vec(shape1);
    for (size_t j = 0; j < shape1; j++) {
      switch (inputType_) {
        case mlprof::InputType::Zeros:
          vec[i] = float(0);
          break;
        case mlprof::InputType::Ones:
          vec[i] = float(1);
          break;
        case mlprof::InputType::Incremental:
          vec[i] = float(j);
          break;
        case mlprof::InputType::Random:
          vec[i] = drawNormalFloat();
          break;
      }
    }
    input.push_back(vec);
  }
  return input;
}

tfaot::DoubleArrays TFAOTInference::createDoubleInput(size_t shape1) {
  tfaot::DoubleArrays input;
  for (int i = 0; i < batchSize_; i++) {
    std::vector<double> vec(shape1);
    for (size_t j = 0; j < shape1; j++) {
      switch (inputType_) {
        case mlprof::InputType::Zeros:
          vec[i] = double(0);
          break;
        case mlprof::InputType::Ones:
          vec[i] = double(1);
          break;
        case mlprof::InputType::Incremental:
          vec[i] = double(j);
          break;
        case mlprof::InputType::Random:
          vec[i] = drawNormalDouble();
          break;
      }
    }
    input.push_back(vec);
  }
  return input;
}

void TFAOTInference::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // prepare inputs
  // INSERT=inputs

  // prepare output vectors
  // INSERT=outputs

  // pre calls to "warm up"
  for (int r = 0; r < nPreCalls_; r++) {
    // INSERT=untied_inference
  }

  // actual calls to measure runtimes
  std::vector<float> runtimes;
  for (int r = 0; r < nCalls_; r++) {
    auto start = std::chrono::high_resolution_clock::now();

    // inference
    // INSERT=tied_inference

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> runtime_in_seconds = (end - start);
    runtimes.push_back(runtime_in_seconds.count() * 1000);
  }

  // save them
  mlprof::writeRuntimes(outputFile_, batchSize_, runtimes);
}

DEFINE_FWK_MODULE(TFAOTInference);
