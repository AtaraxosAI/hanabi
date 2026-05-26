#pragma once

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <vector>
#include <queue>

#include "cpp/infra/tensor_dict.h"
#include "cpp/infra/transition.h"
#include "cpp/infra/concurrent_queue.h"

namespace infra {

class Replay {
 public:
  Replay(int capacity, int seed, int prefetch, float storageRatio = 1.25)
      : prefetch_(prefetch)
      , capacity_(capacity)
      , storage_(int(storageRatio * capacity))
      , numAdd_(0) {
    rng_.seed(seed);
    std::cout << "Create replay with " << storageRatio * capacity << " (" << storageRatio << "x" <<  capacity << ")" << std::endl;
  }

  void clear() {
    // only safe when no prefetch sampler thread is running
    assert(samplerThread_ == nullptr);
    storage_.clear();
    // do not reset numAdd_: it's a monotonic throughput counter used by tachometer
    // wake any waitIfFull caller parked while the buffer was full
    {
      std::lock_guard<std::mutex> lk(mFull_);
    }
    cvFull_.notify_all();
  }

  void waitUntilFull() {
    std::unique_lock<std::mutex> lk(mFull_);
    cvFull_.wait(lk, [this] {
      return terminated_ || storage_.safeSize(nullptr) >= capacity_;
    });
  }

  // get the entire replay as a single batch (no prefetch, no pop)
  // requires the buffer to be full and no sampler thread running
  RNNTransition getAll(const std::string& device, bool shuffle) {
    assert(samplerThread_ == nullptr);
    int size = storage_.safeSize(nullptr);
    assert(size == capacity_);

    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    if (shuffle) {
      std::shuffle(indices.begin(), indices.end(), rng_);
    }

    auto batch = storage_.gather(indices, device);
    batch.seqFirst_();
    return batch;
  }

  void terminate() {
    storage_.terminate();
    // wake any thread parked in waitUntilFull so it can observe shutdown
    {
      std::lock_guard<std::mutex> lk(mFull_);
      terminated_ = true;
    }
    cvFull_.notify_all();
  }

  void add(const RNNTransition& sample) {
    numAdd_ += 1;
    storage_.append(sample, 1);
    if (storage_.safeSize(nullptr) >= capacity_) {
      std::lock_guard<std::mutex> lk(mFull_);
      cvFull_.notify_all();
    }
  }

//   // block until storage has room (safeSize < capacity_) or terminated.
//   // for on-policy training: actor calls this before add() to apply backpressure.
//   void waitIfFull() {
//     std::unique_lock<std::mutex> lk(mFull_);
//     cvFull_.wait(lk, [this] {
//       return terminated_ || storage_.safeSize(nullptr) < capacity_;
//     });
//   }

  RNNTransition sample(int batchsize, const std::string& device) {
    // simple, single thread version
    if (prefetch_ == 0) {
      return sample_(batchsize, device);
    }

    if (samplerThread_ == nullptr) {
      // create sampler thread
      samplerThread_ = std::make_unique<std::thread>(
          &Replay::sampleLoop_, this, batchsize, device);
    }

    // std::cout << "Try to get batch" << std::endl;
    std::unique_lock<std::mutex> lk(mSampler_);
    cvSampler_.wait(lk, [this] {return samples_.size() > 0;});
    // std::cout << "Get Batch" << std::endl;

    auto batch = samples_.front();
    samples_.pop();

    lk.unlock();
    cvSampler_.notify_all();
    // std::cout << "Return Batch" << std::endl;
    return batch;
  }

  RNNTransition get(int idx) {
    return storage_.get(idx);
  }

  RNNTransition getRange(int start, int end, const std::string& device) {
    std::vector<RNNTransition> samples;
    for (int i = start; i < end; ++i) {
      samples.push_back(storage_.get(i));
    };
    return makeBatch(samples, device);
  }

  int size() const {
    return storage_.safeSize(nullptr);
  }

  int numAdd() const {
    return numAdd_;
  }

 private:
  void sampleLoop_(int batchsize, const std::string& device) {
    while (true) {
      auto batch = sample_(batchsize, device);

      std::unique_lock<std::mutex> lk(mSampler_);
      cvSampler_.wait(lk, [this] {return (int)samples_.size() < prefetch_;});
      samples_.push(batch);
      // std::cout << "samples size: " << samples_.size() << std::endl;
      lk.unlock();
      cvSampler_.notify_all();
    }
  }

  RNNTransition sample_(int batchsize, const std::string& device) {
    float sum;
    int size = storage_.safeSize(&sum);
    assert(int(sum) == size);
    assert(size >= batchsize);
    // storage_ [0, size) remains static in the subsequent section
    int segment = size / batchsize;
    std::uniform_int_distribution<int> dist(0, segment-1);

    assert(batchsize > 0);
    RNNTransition batch(storage_.get(0), batchsize);

    // RNNTransition batch;
    for (int i = 0; i < batchsize; ++i) {
      int rand = dist(rng_) + i * segment;
      assert(rand < size);
      storage_.copyTo(rand, batch, i);
    }

    // pop storage if at/above capacity. must leave safeSize < capacity_ so
    // waitIfFull (predicate: safeSize < capacity_) can make progress —
    // popping only when size > capacity_ deadlocks at exactly size == capacity_.
    size = storage_.size();
    if (size >= capacity_) {
      storage_.blockPop(size - capacity_ + 1);
      // wake any blockingAdd callers parked because storage was full
      {
        std::lock_guard<std::mutex> lk(mFull_);
      }
      cvFull_.notify_all();
    }
    batch.to_(device);
    batch.seqFirst_();
    return batch;
  }

  const int prefetch_;
  const int capacity_;

  // make sure that multiple calls of sample does not overlap
  std::unique_ptr<std::thread> samplerThread_;
  // basic concurrent queue for read and write data
  std::queue<RNNTransition> samples_;
  std::mutex mSampler_;
  std::condition_variable cvSampler_;

  // signals waitUntilFull when storage reaches capacity or on terminate
  std::mutex mFull_;
  std::condition_variable cvFull_;
  bool terminated_ = false;

  ConcurrentQueue storage_;
  std::atomic<int> numAdd_;

  std::mt19937 rng_;
};

}
