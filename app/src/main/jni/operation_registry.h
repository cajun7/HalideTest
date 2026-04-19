#pragma once

#include "operation.h"
#include <memory>
#include <vector>
#include <cstring>

// Singleton registry for image processing operations.
// Operations self-register via static initializers in their .cpp files.
//
// Pattern from Halide's RunGen tool and Android Camera HAL's block registry.
//
// Usage in op files:
//   static auto _reg = [] {
//       OperationRegistry::instance().add(std::make_unique<MyOp>());
//       return 0;
//   }();

class OperationRegistry {
public:
    // Meyer's singleton — guaranteed thread-safe init in C++11+
    static OperationRegistry& instance() {
        static OperationRegistry reg;
        return reg;
    }

    void add(std::unique_ptr<IOperation> op) {
        ops_.push_back(std::move(op));
    }

    IOperation* find(const char* name) const {
        for (auto& op : ops_) {
            if (std::strcmp(op->name(), name) == 0) {
                return op.get();
            }
        }
        return nullptr;
    }

    const std::vector<std::unique_ptr<IOperation>>& all() const {
        return ops_;
    }

    size_t size() const { return ops_.size(); }

private:
    OperationRegistry() = default;
    std::vector<std::unique_ptr<IOperation>> ops_;
};

// Convenience macro for self-registration
#define REGISTER_OP(ClassName) \
    static int _reg_##ClassName = [] { \
        OperationRegistry::instance().add(std::make_unique<ClassName>()); \
        return 0; \
    }()
