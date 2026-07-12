/**
 * @file dlpack_python.cpp
 * @brief Python-specific DLPack wrapper functions
 *
 * Separated from dlpack_bridge.cpp to avoid Python.h dependency in core library.
 */

#include <Python.h>
#include "dlpack_bridge.hpp"

namespace mcts {

// Include DLPack type definitions (duplicated from dlpack_bridge.cpp)
extern "C" {

typedef enum {
    kDLCPU = 1,
    kDLCUDA = 2,
    kDLCUDAHost = 3,
    kDLCUDAManaged = 13,
} DLDeviceType;

typedef struct {
    int device_type;
    int device_id;
} DLDevice;

typedef enum {
    kDLFloat = 2,
    kDLUInt = 1,
    kDLInt = 0,
} DLDataTypeCode;

typedef struct {
    uint8_t code;
    uint8_t bits;
    uint16_t lanes;
} DLDataType;

typedef struct DLTensor {
    void* data;
    DLDevice device;
    int32_t ndim;
    DLDataType dtype;
    int64_t* shape;
    int64_t* strides;
    uint64_t byte_offset;
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

} // extern "C"

// Wrap DLManagedTensor in PyCapsule for Python
PyObject* wrap_dlpack_capsule(DLManagedTensor* tensor) {
    if (!tensor) {
        PyErr_SetString(PyExc_RuntimeError, "wrap_dlpack_capsule: tensor is null");
        return nullptr;
    }

    // Create PyCapsule with "dltensor" name (required by torch.from_dlpack)
    // Note: We do NOT provide a capsule destructor because PyTorch calls the
    // DLManagedTensor deleter itself after consumption. The capsule is just
    // a transport mechanism.
    PyObject* capsule = PyCapsule_New(
        tensor,
        "dltensor",
        nullptr  // No capsule destructor - PyTorch handles it
    );

    if (!capsule) {
        // Failed to create capsule, clean up tensor
        dlpack_deleter(tensor);
        return nullptr;
    }

    return capsule;
}

} // namespace mcts
