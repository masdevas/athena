#ifndef ATHENA_OPENCLDEVICE_H
#define ATHENA_OPENCLDEVICE_H

#include "BufferAllocator.h"
#include "OpenClQueue.h"

#include <athena/backend/llvm/runtime/Device.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace athena::backend::llvm {
class OpenCLDevice : public Device {
public:
  explicit OpenCLDevice(cl_device_id id) : mClDeviceId(id) {
    size_t strLength;
    clGetDeviceInfo(mClDeviceId, CL_DEVICE_NAME, 0, nullptr, &strLength);
    mDeviceName.resize(strLength);
    clGetDeviceInfo(mClDeviceId, CL_DEVICE_NAME, strLength, mDeviceName.data(),
                    nullptr);

    // fixme check for errors
    mContext =
        clCreateContext(nullptr, 1, &mClDeviceId, nullptr, nullptr, nullptr);

    mQueue = std::make_unique<OpenClQueue>(mContext, mClDeviceId);
    // todo support USM allocator as well.
    mAllocator = std::make_shared<BufferAllocator>(mContext);
  }

  Queue& getQueue() override { return *mQueue; }

  std::string getDeviceName() const override { return mDeviceName; }
  bool isPartitionSupported(PartitionDomain domain) override {
    return false; // todo implement
  }
  DeviceContainer partition(PartitionDomain domain) override {
    return Device::partition(domain); // todo implement
  }
  bool hasAllocator() override { return true; }
  std::shared_ptr<AllocatorLayerBase> getAllocator() override {
    return mAllocator;
  }
  bool operator==(const Device& device) const override {
    // fixme it must compare device ids.
    return mDeviceName == device.getDeviceName();
  }
  void copyToHost(const core::inner::Tensor& tensor,
                  void* dest) const override {
    MemoryRecord record{tensor.getVirtualAddress(),
                        tensor.getShapeView().getTotalSize() *
                            core::sizeOfDataType(tensor.getDataType())};
    copyToHost(record, dest);
  }
  void copyToHost(MemoryRecord record, void* dest) const override {
    auto buf = *static_cast<cl_mem*>(mAllocator->getPtr(record));
    clEnqueueReadBuffer(mQueue->getNativeQueue(), buf, CL_TRUE, 0, // offset
                        record.allocationSize, dest,
                        0, // num events
                        nullptr, nullptr);
  }
  void copyToDevice(const core::inner::Tensor& tensor,
                    void* src) const override {
    MemoryRecord record{tensor.getVirtualAddress(),
                        tensor.getShapeView().getTotalSize() *
                            core::sizeOfDataType(tensor.getDataType())};
    copyToDevice(record, src);
  }
  void copyToDevice(MemoryRecord record, void* src) const override {
    auto buf = *static_cast<cl_mem*>(mAllocator->getPtr(record));
    clEnqueueWriteBuffer(mQueue->getNativeQueue(), buf, CL_TRUE, 0, // offset
                        record.allocationSize, src,
                        0, // num events
                        nullptr, nullptr);
  }

  cl_device_id getNativeDevice() { return mClDeviceId; }

  cl_context getContext() { return mContext; }

  void addProgram(cl_program program) { mPrograms.push_back(program); }

  const std::vector<cl_program> getPrograms() const { return mPrograms; }

  void setLinkedProgram(cl_program program) { mLinkedProgram = program; }
  cl_program getLinkedProgram() { return mLinkedProgram; }

private:
  cl_device_id mClDeviceId;
  cl_context mContext;
  std::string mDeviceName;
  std::vector<cl_program> mPrograms;
  cl_program mLinkedProgram;
  std::unique_ptr<OpenClQueue> mQueue;
  std::shared_ptr<BufferAllocator> mAllocator;
};
} // namespace athena::backend::llvm

#endif // ATHENA_OPENCLDEVICE_H