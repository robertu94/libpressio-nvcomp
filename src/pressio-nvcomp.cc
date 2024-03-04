#include <map>
#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/pressio.h>
#include <std_compat/memory.h>
#include <std_compat/functional.h>
#include <sstream>
#include <cuda.h>
#include <nvcomp.hpp>
#include <nvcomp/nvcompManagerFactory.hpp>
#include <nvcomp/cascaded.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/snappy.hpp>
#include <nvcomp/lz4.hpp>

extern "C" void libpressio_register_nvcomp() {
}

namespace libpressio { namespace nvcomp_ns {

  /**
 * this class is a standard c++ idiom for closing resources
 * it calls the function passed in during the destructor.
 */
class cleanup {
  public:
    cleanup() noexcept: cleanup_fn([]{}), do_cleanup(false) {}

    template <class Function>
    cleanup(Function f) noexcept: cleanup_fn(std::forward<Function>(f)), do_cleanup(true) {}
    cleanup(cleanup&& rhs) noexcept: cleanup_fn(std::move(rhs.cleanup_fn)), do_cleanup(compat::exchange(rhs.do_cleanup, false)) {}
    cleanup(cleanup const&)=delete;
    cleanup& operator=(cleanup const&)=delete;
    cleanup& operator=(cleanup && rhs) noexcept { 
      if(&rhs == this) return *this;
      do_cleanup = compat::exchange(rhs.do_cleanup, false);
      cleanup_fn = std::move(rhs.cleanup_fn);
      return *this;
    }
    ~cleanup() { if(do_cleanup) cleanup_fn(); }

  private:
    std::function<void()> cleanup_fn;
    bool do_cleanup;
};
template<class Function>
cleanup make_cleanup(Function&& f) {
  return cleanup(std::forward<Function>(f));
}


enum lp_nv_exec_mode{
  LP_NV_CASCADED = 0,
  LP_NV_LZ4 = 1,
  LP_NV_SNAPPY = 2,
  LP_NV_BITCOMP = 3,
  LP_NV_GDEFLATE = 4,
};

std::map<std::string, int> lp_nv_exec_mode_map {
  {"cascade", LP_NV_CASCADED},
  {"lz4", LP_NV_LZ4},
  {"snappy", LP_NV_SNAPPY},
  {"bitcomp", LP_NV_BITCOMP},
  {"gdeflate", LP_NV_GDEFLATE},
};

class pressio_cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
void pressio_cuda_check(cudaError err) {
  if(err != cudaSuccess) {
    throw pressio_cuda_error(cudaGetErrorString(err));
  }
}

class pressio_nvcomp: public libpressio_compressor_plugin {
  private:
  pressio_options get_options_impl() const override {
    pressio_options opts;
    set(opts, "nvcomp:chunk_size", chunk_size);
    set(opts, "nvcomp:device", device);
    set(opts, "nvcomp:num_rles", num_rles);
    set(opts, "nvcomp:num_deltas", num_deltas);
    set(opts, "nvcomp:use_bp", use_bp);
    set(opts, "nvcomp:alg", alg);
    set_type(opts, "nvcomp:alg_str", pressio_option_charptr_type);
    set(opts, "nvcomp:nvcomp_alg", nvcomp_alg);
    set_type(opts, "pressio:lossless", pressio_option_int32_type);

    return opts;
  }
  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", R"(nvcomp

    libpressio bindings for the NVIDIA nvcomp GPU compressors
    )");
    set(opts, "nvcomp:chunk_size", "chunk size: default 4096, valid values are 512 to 16384");
    set(opts, "nvcomp:device", "nvidia device to execute on");
    set(opts, "nvcomp:num_rles", "number of run length encoding to preform");
    set(opts, "nvcomp:num_deltas", "number of delta encodings to preform");
    set(opts, "nvcomp:use_bp", "preform bit comp as the last step");
    set(opts, "nvcomp:alg", "which algorithm to use");
    set(opts, "nvcomp:alg_str", "which algorithm to use as a string");
    set(opts, "nvcomp:nvcomp_alg", "for nvcomp algorithms that support it, which variant to use");
    return opts;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "pressio:stability", "experimental");
    set(opts, "pressio:lossless:min", 0);
    set(opts, "pressio:lossless:max", 0);
    std::vector<std::string> alg_strings;
    for (auto const& i : lp_nv_exec_mode_map) {
       alg_strings.push_back(i.first); 
    }
    set(opts, "nvcomp:alg_str", alg_strings);

    
        std::vector<std::string> invalidations {"nvcomp:chunk_size", "nvcomp:device", "nvcomp:num_rles", "nvcomp:num_deltas", "nvcomp:use_bp", "nvcomp:alg", "nvcomp:alg_str", "nvcomp:nvcomp_alg"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(opts, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, {}));
        set(opts, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(opts, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));

    return opts;
  }
  int set_options_impl(const pressio_options &opts) override {
    get(opts, "nvcomp:chunk_size", &chunk_size);
    get(opts, "nvcomp:device", &device);
    get(opts, "nvcomp:num_rles", &num_rles);
    get(opts, "nvcomp:num_deltas", &num_deltas);
    get(opts, "nvcomp:use_bp", &use_bp);
    bool alg_set = false;
    int32_t tmp;
    if(get(opts, "nvcomp:alg", &tmp) == pressio_options_key_set) {
      if(tmp >= 0 && tmp <= LP_NV_GDEFLATE) {
        alg = tmp;
        alg_set = true;
      } else {
        return set_error(1, "unsupported alg_str " + std::to_string(tmp));
      }
    }
    std::string tmp_s;
    if(get(opts, "nvcomp:alg_str", &tmp_s) == pressio_options_key_set) {
      std::map<std::string, int>::iterator it;
      if((it = lp_nv_exec_mode_map.find(tmp_s)) != lp_nv_exec_mode_map.end()) {
        alg = it->second;
        alg_set = true;
      } else {
        return set_error(1, "unsupported alg_str " + tmp_s);
      }
    }

    
    tmp = 0;
    pressio_options_key_status st;
    if((st = get(opts, "nvcomp:nvcomp_alg", &tmp)) == pressio_options_key_set  || alg_set) {
      switch(alg) {
        case LP_NV_SNAPPY:
        case LP_NV_LZ4:
        case LP_NV_CASCADED:
          nvcomp_alg = 0;
          break;
        case LP_NV_GDEFLATE:
          if(tmp >= 0 && tmp <= 2) {
            nvcomp_alg = tmp;
          } else {
            nvcomp_alg = 0;
          }
        case LP_NV_BITCOMP:
          if(tmp >= 0 && tmp <= 1) {
            nvcomp_alg = tmp;
          } else {
            nvcomp_alg = 0;
          }
      }
    }
    return 0;
  }

  std::unique_ptr<nvcomp::nvcompManagerBase> get_manager(cudaStream_t stream, pressio_dtype dtype) {
    if(dtype != pressio_byte_dtype) {
      throw std::runtime_error("unsupported type");
    }
    nvcompType_t t = NVCOMP_TYPE_CHAR;
    switch(alg) {
      case LP_NV_CASCADED:
        return compat::make_unique<nvcomp::CascadedManager>(nvcompBatchedCascadedOpts_t{
              static_cast<size_t>(chunk_size),
              t,
              num_rles,
              num_deltas,
              use_bp
            }, stream, device);
      case LP_NV_BITCOMP:
        return compat::make_unique<nvcomp::BitcompManager>(t, nvcomp_alg, stream, device);
      case LP_NV_GDEFLATE:
        return compat::make_unique<nvcomp::GdeflateManager>(t, nvcomp_alg, stream, device);
      case LP_NV_SNAPPY:
        return compat::make_unique<nvcomp::SnappyManager>(chunk_size, stream, device);
      case LP_NV_LZ4:
        return compat::make_unique<nvcomp::LZ4Manager>(chunk_size, t, stream, device);
    }
    throw std::runtime_error("unsupported compression algorithm");
  }


  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    try {
      cudaStream_t stream;
      pressio_cuda_check(cudaStreamCreate(&stream));
      auto cleanup_stream = make_cleanup([&stream]{ cudaStreamDestroy(stream); });

      auto mgr = get_manager(stream, pressio_byte_dtype);
      nvcomp::CompressionConfig cfg = mgr->configure_compression(input->size_in_bytes());

      uint8_t* comp_buffer;
      pressio_cuda_check(cudaMalloc(&comp_buffer, cfg.max_compressed_buffer_size));
      auto cleanup_comp = cleanup([comp_buffer]{cudaFree(comp_buffer);});

      uint8_t* input_buffer;
      pressio_cuda_check(cudaMalloc(&input_buffer, input->size_in_bytes()));
      auto cleanup_input= make_cleanup([input_buffer]{cudaFree(input_buffer);});
      pressio_cuda_check(cudaMemcpy(input_buffer, input->data(), input->size_in_bytes(), cudaMemcpyHostToDevice));

      mgr->compress(input_buffer, comp_buffer, cfg);

      size_t comp_size = mgr->get_compressed_output_size(comp_buffer);

      if(output->capacity_in_bytes() < comp_size) {
        *output = pressio_data::owning(pressio_byte_dtype, {comp_size});
      } else {
        output->set_dtype(pressio_byte_dtype);
        output->set_dimensions({comp_size});
      }

      pressio_cuda_check(cudaMemcpy(output->data(), comp_buffer, comp_size, cudaMemcpyDeviceToHost));
      pressio_cuda_check(cudaStreamSynchronize(stream));
    } catch(std::runtime_error const& ex) {
      return set_error(1, ex.what());
    }

     return 0;
  }
  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    try {
      cudaStream_t stream;
      pressio_cuda_check(cudaStreamCreate(&stream));
      auto cleanup_stream = make_cleanup([&stream]{ cudaStreamDestroy(stream); });
      uint8_t* comp_buffer;
      pressio_cuda_check(cudaMalloc(&comp_buffer, input->size_in_bytes()));
      auto cleanup_comp = cleanup([comp_buffer]{cudaFree(comp_buffer);});
      pressio_cuda_check(cudaMemcpy(comp_buffer, input->data(), input->size_in_bytes(), cudaMemcpyHostToDevice));

      auto mgr = nvcomp::create_manager(comp_buffer, stream, device);
      nvcomp::DecompressionConfig cfg = mgr->configure_decompression(comp_buffer);

      uint8_t* out_buffer;
      pressio_cuda_check(cudaMalloc(&out_buffer, cfg.decomp_data_size));
      auto cleanup_out = make_cleanup([out_buffer]{cudaFree(out_buffer);});

      mgr->decompress(out_buffer, comp_buffer, cfg);
      
      pressio_cuda_check(cudaMemcpy(output->data(), out_buffer, output->size_in_bytes(), cudaMemcpyDeviceToHost));

      pressio_cuda_check(cudaStreamSynchronize(stream));
    } catch(std::runtime_error const& ex) {
      return set_error(1, ex.what());
    }
    return 0;
  }

  int major_version() const override {
    return NVCOMP_MAJOR_VERSION;
  }
  int minor_version() const override {
    return NVCOMP_MINOR_VERSION;
  }
  int patch_version() const override {
    return NVCOMP_PATCH_VERSION;
  }

  void set_name_impl(std::string const& new_name) override {
  }
  const char* prefix() const override { return "nvcomp"; }
  const char* version() const override { 
    const static std::string version_str = [this]{
      std::stringstream ss;
      ss << major_version() << '.' << minor_version() << '.' << patch_version();
      return ss.str();
    }();
    return version_str.c_str();
  }
  virtual std::shared_ptr<libpressio_compressor_plugin> clone() override {
    return compat::make_unique<pressio_nvcomp>(*this);
  }

  int32_t chunk_size = 1<<16;
  int32_t device = 0;
  int32_t num_rles = 2;
  int32_t num_deltas = 1;
  int32_t use_bp = 1;
  int32_t alg = LP_NV_CASCADED;
  int32_t nvcomp_alg = 0;
};

static pressio_register pressio_nvcomp_register(
    compressor_plugins(),
    "nvcomp",
    []{
      return compat::make_unique<pressio_nvcomp>();
    }
    );

} }
