// Native cuBLASLt selection only: no new GPU arithmetic or precision mode.
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <cublasLt.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
constexpr int64_t M = 1536, N = 1536, K = 32768;
constexpr int MaxChoices = 6;
struct Counts {
  std::atomic<uint64_t> matched{0}, unsupported{0}, capture{0}, cached{0},
      tuned{0}, native{0}, dispatched{0};
} counts;

void check(cublasStatus_t status) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS,
              "nanochat cuBLASLt error ", static_cast<int>(status));
}
void log(const std::string& message) {
  const auto* rank = std::getenv("RANK");
  std::ostringstream line;
  line << "NANOCHAT_OPT_LT_SEARCH: rank=" << (rank ? rank : "unset")
       << " pid=" << getpid() << " " << message << '\n';
  std::cout << line.str() << std::flush;
}
struct MatmulDesc {
  cublasLtMatmulDesc_t value = nullptr;
  MatmulDesc() { check(cublasLtMatmulDescCreate(&value, CUBLAS_COMPUTE_32F, CUDA_R_32F)); }
  MatmulDesc(const MatmulDesc&) = delete;
  MatmulDesc& operator=(const MatmulDesc&) = delete;
  ~MatmulDesc() { if (value) cublasLtMatmulDescDestroy(value); }
  template<class T> void set(cublasLtMatmulDescAttributes_t attribute, T v) {
    check(cublasLtMatmulDescSetAttribute(value, attribute, &v, sizeof(v)));
  }
};
struct Layout {
  cublasLtMatrixLayout_t value = nullptr;
  Layout(cudaDataType_t type, int64_t rows, int64_t columns, int64_t ld) {
    check(cublasLtMatrixLayoutCreate(&value, type, rows, columns, ld));
  }
  ~Layout() { if (value) cublasLtMatrixLayoutDestroy(value); }
  Layout(const Layout&) = delete;
  Layout& operator=(const Layout&) = delete;
};
struct Preference {
  cublasLtMatmulPreference_t value = nullptr;
  Preference() { check(cublasLtMatmulPreferenceCreate(&value)); }
  void workspace(size_t bytes) {
    check(cublasLtMatmulPreferenceSetAttribute(
        value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &bytes, sizeof(bytes)));
  }
  ~Preference() { if (value) cublasLtMatmulPreferenceDestroy(value); }
  Preference(const Preference&) = delete;
  Preference& operator=(const Preference&) = delete;
};
struct Event {
  cudaEvent_t value = nullptr;
  Event() { C10_CUDA_CHECK(cudaEventCreateWithFlags(&value, cudaEventDefault)); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() { if (value) cudaEventDestroy(value); }
};

bool shape(const at::Tensor& t, at::ScalarType dtype,
           int64_t rows, int64_t columns, int64_t s0, int64_t s1) {
  return t.is_cuda() && t.layout() == at::kStrided && t.scalar_type() == dtype &&
      t.dim() == 2 && t.size(0) == rows && t.size(1) == columns &&
      t.stride(0) == s0 && t.stride(1) == s1 && !t.is_conj() && !t.is_neg();
}
bool scalar(const at::Tensor& t) {
  return t.is_cuda() && t.layout() == at::kStrided &&
      t.scalar_type() == at::kFloat && t.dim() == 0 &&
      !t.is_conj() && !t.is_neg();
}
bool disjoint(const at::Tensor& output, const at::Tensor& input) {
  // Every supported tensor is dense. data_ptr includes its storage offset.
  const auto o = reinterpret_cast<uintptr_t>(output.data_ptr());
  const auto i = reinterpret_cast<uintptr_t>(input.data_ptr());
  const auto on = static_cast<uintptr_t>(output.numel() * output.element_size());
  const auto in = static_cast<uintptr_t>(input.numel() * input.element_size());
  // Difference form avoids endpoint overflow.
  return o <= i ? i - o >= on : o - i >= in;
}
bool tensors_ok(const at::Tensor& a, const at::Tensor& b,
                const at::Tensor& sa, const at::Tensor& sb, const at::Tensor& out) {
  if (!shape(a, at::kFloat8_e5m2, M, K, K, 1) ||
      !shape(b, at::kFloat8_e4m3fn, K, N, 1, K) ||
      !shape(out, at::kBFloat16, M, N, N, 1) || !scalar(sa) || !scalar(sb))
    return false;
  for (const auto* t : {&b, &sa, &sb, &out})
    if (t->device() != a.device()) return false;
  for (const auto* t : {&a, &b, &out})
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % 256) return false;
  for (const auto* t : {&sa, &sb})
    if (reinterpret_cast<uintptr_t>(t->data_ptr()) % alignof(float)) return false;
  for (const auto* t : {&a, &b, &sa, &sb})
    if (!disjoint(out, *t)) return false;
  return true;
}

struct Attributes {
  uint32_t id = 0, split = 0, reduction = 0;
  uint64_t flags = 0;
  std::array<uint32_t, 4> alignment{};
  // cuBLASLt permits serialization of this opaque descriptor within the same
  // library version. This copy is for audit logging only, never selection.
  std::array<unsigned char, sizeof(cublasLtMatmulAlgo_t)> raw{};
};
template<class T> bool config_attr(const cublasLtMatmulAlgo_t& algo,
                                  cublasLtMatmulAlgoConfigAttributes_t name, T& value) {
  size_t written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(
      &algo, name, &value, sizeof(value), &written) == CUBLAS_STATUS_SUCCESS &&
      written == sizeof(value);
}
template<class T> bool cap_attr(const cublasLtMatmulAlgo_t& algo,
                               cublasLtMatmulAlgoCapAttributes_t name, T& value) {
  size_t written = 0;
  return cublasLtMatmulAlgoCapGetAttribute(
      &algo, name, &value, sizeof(value), &written) == CUBLAS_STATUS_SUCCESS &&
      written == sizeof(value);
}
bool attributes(const cublasLtMatmulAlgo_t& algo, Attributes& info) {
  std::memcpy(info.raw.data(), &algo, info.raw.size());
  if (!config_attr(algo, CUBLASLT_ALGO_CONFIG_ID, info.id) ||
      !config_attr(algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, info.split) ||
      !config_attr(algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, info.reduction) ||
      !cap_attr(algo, CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS, info.flags))
    return false;
  const cublasLtMatmulAlgoCapAttributes_t names[] = {
      CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES,
      CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES};
  for (size_t i = 0; i < 4; ++i)
    if (!cap_attr(algo, names[i], info.alignment[i])) return false;
  return true;
}
bool precision_ok(const Attributes& info) {
  if (info.split > 1 ||
      (info.flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_TYPE_MASK) !=
          CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F)
    return false;
  // Pointers and all leading dimensions in bytes are multiples of 256.
  for (auto alignment : info.alignment)
    if (!alignment || alignment > 256 || 256 % alignment) return false;
  return true;
}
std::string describe(const Attributes& a) {
  std::ostringstream s;
  s << "{\"id\":" << a.id << ",\"split\":" << a.split
    << ",\"reduction\":" << a.reduction << ",\"flags\":" << a.flags
    << ",\"alignment\":[" << a.alignment[0] << "," << a.alignment[1]
    << "," << a.alignment[2] << "," << a.alignment[3] << "],\"algorithm_hex\":\"";
  for (auto byte : a.raw)
    s << std::hex << std::setfill('0') << std::setw(2) << static_cast<unsigned>(byte);
  s << "\"}";
  return s.str();
}

struct Key {
  int device;
  std::array<char, 16> uuid;
  size_t library_version, workspace;
  bool operator==(const Key& other) const {
    return device == other.device && uuid == other.uuid &&
        library_version == other.library_version && workspace == other.workspace;
  }
  // Fixed shape/strides/dtypes/scalar modes/fast-accum/alignment are implicit
  // in this source-bound ABI and are checked on every call.
};
struct Entry {
  Key key;
  bool optimized = false;
  cublasLtMatmulAlgo_t algo{};
};
thread_local std::vector<Entry> cache;

bool run(const at::Tensor& a, const at::Tensor& b, const at::Tensor& sa,
         const at::Tensor& sb, const at::Tensor& out) {
  if (!tensors_ok(a, b, sa, sb, out)) { ++counts.unsupported; return false; }
  ++counts.matched;
  c10::cuda::CUDAGuard guard(a.device());
  const auto stream = c10::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone) { ++counts.capture; return false; }
  if (at::globalContext()._SMCarveout_EXPERIMENTAL().has_value()) {
    ++counts.unsupported; return false;
  }
  const auto* props = at::cuda::getCurrentDeviceProperties();
  if (props->major != 9 || props->minor != 0 || props->multiProcessorCount != 78) {
    ++counts.unsupported; return false;
  }
  Key key{a.get_device(), {}, cublasLtGetVersion(), at::cuda::getCUDABlasLtWorkspaceSize()};
  std::memcpy(key.uuid.data(), props->uuid.bytes, key.uuid.size());
  auto entry = std::find_if(cache.begin(), cache.end(), [&](const Entry& e) { return e.key == key; });
  if (entry != cache.end()) {
    ++counts.cached;
    if (!entry->optimized) { ++counts.native; return false; }
  }

  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto* workspace = at::cuda::getCUDABlasLtWorkspace();
  MatmulDesc desc;
  desc.set(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T);
  desc.set(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N);
  // Native cublasCommonArgs computes row-major output as B.T @ A.T.
  // Swap operands and scalar pointers, retaining column-major descriptors.
  desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, sb.data_ptr());
  desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, sa.data_ptr());
  desc.set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, int8_t{0});
  desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
  desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F);
  Layout ad(CUDA_R_8F_E4M3, K, N, K), bd(CUDA_R_8F_E5M2, K, M, K);
  Layout cd(CUDA_R_16BF, N, M, N), dd(CUDA_R_16BF, N, M, N);
  const float alpha = 1.0f, beta = 0.0f;
  auto launch = [&](const cublasLtMatmulAlgo_t& algorithm) {
    // Never convert a CUDA/library execution failure into a native fallback.
    check(cublasLtMatmul(handle, desc.value, &alpha, b.data_ptr(), ad.value,
                        a.data_ptr(), bd.value, &beta, nullptr, cd.value,
                        out.data_ptr(), dd.value, &algorithm,
                        workspace, key.workspace, stream));
  };
  if (entry == cache.end()) {
    Entry selected{key};
    auto native = [&](const std::string& reason) {
      log("native fallback: " + reason);
      cache.push_back(selected);
      ++counts.native;
      return false;
    };
    Preference pref;
    pref.workspace(key.workspace);
    cublasLtMatmulHeuristicResult_t baseline{};
    int returned = 0;
    auto status = cublasLtMatmulAlgoGetHeuristic(
        handle, desc.value, ad.value, bd.value, cd.value, dd.value,
        pref.value, 1, &baseline, &returned);
    if (status != CUBLAS_STATUS_SUCCESS || returned != 1 ||
        baseline.state != CUBLAS_STATUS_SUCCESS) return native("baseline heuristic unavailable");
    Attributes base_info;
    if (!attributes(baseline.algo, base_info)) return native("baseline attributes unknown");
    log("baseline " + describe(base_info) + " workspace=" + std::to_string(key.workspace) +
        " library=" + std::to_string(key.library_version));
    if (!precision_ok(base_info)) return native("baseline outside unsplit FP32 contract");
    auto legal = [&](const cublasLtMatmulAlgo_t& algo) {
      cublasLtMatmulHeuristicResult_t checked{};
      return cublasLtMatmulAlgoCheck(handle, desc.value, ad.value, bd.value,
          cd.value, dd.value, &algo, &checked) == CUBLAS_STATUS_SUCCESS &&
          checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize <= key.workspace;
    };
    if (!legal(baseline.algo)) return native("baseline AlgoCheck unavailable");
    std::vector<cublasLtMatmulAlgo_t> choices{baseline.algo};
    std::vector<Attributes> infos{base_info};
    std::array<cublasLtMatmulHeuristicResult_t, MaxChoices> proposed{};
    returned = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, desc.value, ad.value, bd.value, cd.value, dd.value,
        pref.value, MaxChoices, proposed.data(), &returned);
    if (status != CUBLAS_STATUS_SUCCESS) return native("alternative heuristics unavailable");
    TORCH_CHECK(returned >= 0 && returned <= MaxChoices, "invalid heuristic count");
    for (int i = 0; i < returned && choices.size() < MaxChoices; ++i) {
      const auto& candidate = proposed[i];
      if (candidate.state != CUBLAS_STATUS_SUCCESS) continue;
      Attributes info;
      if (!attributes(candidate.algo, info)) { log("reject unknown attributes"); continue; }
      log("proposed " + describe(info));
      if (!precision_ok(info) || info.flags != base_info.flags ||
          info.reduction != base_info.reduction || !legal(candidate.algo)) {
        log("reject precision/alignment/AlgoCheck"); continue;
      }
      if (std::any_of(choices.begin(), choices.end(), [&](const auto& old) {
            return std::memcmp(&old, &candidate.algo, sizeof(old)) == 0;
          })) continue;
      choices.push_back(candidate.algo);
      infos.push_back(info);
    }
    if (choices.size() < 2) return native("no distinct legal alternatives");
    ++counts.tuned;
    Event start, finish;
    std::vector<float> times;
    for (size_t i = 0; i < choices.size(); ++i) {
      launch(choices[i]);  // Warm the library kernel before measuring it.
      C10_CUDA_CHECK(cudaEventRecord(start.value, stream));
      for (int repeat = 0; repeat < 3; ++repeat) launch(choices[i]);
      C10_CUDA_CHECK(cudaEventRecord(finish.value, stream));
      C10_CUDA_CHECK(cudaEventSynchronize(finish.value));
      float elapsed = 0.0f;
      C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.value, finish.value));
      TORCH_CHECK(std::isfinite(elapsed) && elapsed > 0.0f, "invalid Lt timing result");
      times.push_back(elapsed / 3.0f);
      log("measured " + describe(infos[i]) + " index=" + std::to_string(i) +
          " ms=" + std::to_string(times.back()));
    }
    auto best = static_cast<size_t>(std::min_element(times.begin(), times.end()) - times.begin());
    // Require an isolated-kernel margin before changing the native algorithm.
    if (best == 0 || times[best] >= 0.98f * times[0]) return native("no two-percent tuning margin");
    selected.optimized = true;
    selected.algo = choices[best];
    log("selected " + describe(infos[best]) + " index=" + std::to_string(best) +
        " baseline_ms=" + std::to_string(times[0]) +
        " selected_ms=" + std::to_string(times[best]));
    cache.push_back(selected);
    entry = std::prev(cache.end());
  }
  // Restore the chosen result after all tuning writes; inputs never overlap out.
  launch(entry->algo);
  ++counts.dispatched;
  return true;
}
std::map<std::string, uint64_t> counters() {
  return {{"matched", counts.matched.load()}, {"unsupported", counts.unsupported.load()},
          {"capture", counts.capture.load()}, {"cached", counts.cached.load()},
          {"tuned", counts.tuned.load()}, {"native", counts.native.load()},
          {"dispatched", counts.dispatched.load()}};
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Module initialization registers CPU bindings only. No CUDA calls here.
  m.def("run", &run);
  m.def("counters", &counters);
}
