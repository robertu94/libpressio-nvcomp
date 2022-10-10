#include <libpressio_ext/cpp/compressor.h>
#include <libpressio_ext/cpp/pressio.h>
#include <std_compat/memory.h>
#include <sstream>
#include <cuda.h>
#include <nvcomp.hpp>
#include <nvcomp/cascaded.hpp>

extern "C" void libpressio_register_nvcomp() {
}

class pressio_cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};
void pressio_cuda_check(cudaError err) {
  if(err != cudaSuccess) {
    throw pressio_cuda_error(cudaGetErrorString(err));
  }
}

struct compress_impl_typed {
  template <class T>
  pressio_data operator()(T const* begin, T const* end) {
    std::unique_ptr<nvcomp::CascadedCompressor> c;
    void* d_temp = nullptr;

    //assume data is on the CPU for now
    const size_t in_bytes = std::distance(begin, end);
    T* in_space;
    pressio_cuda_check(cudaMalloc(&in_space, in_bytes));
    pressio_cuda_check(cudaMemcpy(in_space, begin, in_bytes, cudaMemcpyHostToDevice));

    if (g_fopts) {
      //use full custom
      c = compat::make_unique<nvcomp::CascadedCompressor>(
        nvcomp::TypeOf<T>(),
        g_fopts->num_RLEs,
        g_fopts->num_deltas,
        g_fopts->use_bp
        );
    } else if(g_opts) {
      //customize the search
      nvcomp::CascadedSelector<T> selector(in_space, in_bytes, *g_opts);

      size_t temp_size = selector.get_temp_size();
      pressio_cuda_check(cudaMalloc(&d_temp, temp_size));

      double estimate_ratio;
      nvcompCascadedFormatOpts fopts = selector.select_config(d_temp, temp_size, &estimate_ratio, stream);

      c = compat::make_unique<nvcomp::CascadedCompressor>(
        nvcomp::TypeOf<T>(),
        fopts.num_RLEs,
        fopts.num_deltas,
        fopts.use_bp
        );
    } else {
      //use full auto
      c = compat::make_unique<nvcomp::CascadedCompressor>(
        nvcomp::TypeOf<T>() 
        );
    }



    size_t temp_bytes, output_bytes;
    c->configure(in_bytes, &temp_bytes, &output_bytes);
    void *temp_space, *output_space;
    pressio_cuda_check(cudaMalloc(&temp_space, temp_bytes));
    pressio_cuda_check(cudaMalloc(&output_space, output_bytes));

    size_t* d_compresed_bytes;
    pressio_cuda_check(cudaMallocHost(&d_compresed_bytes, sizeof(size_t)));

    c->compress_async(
        in_space, in_bytes,
        temp_space, temp_bytes,
        output_space, d_compresed_bytes,
        stream
        );

    pressio_cuda_check(cudaStreamSynchronize(stream));
    uint8_t* host_buffer = static_cast<uint8_t*>(malloc(*d_compresed_bytes));

    pressio_cuda_check(cudaMemcpy(
        host_buffer,
        output_space,
        *d_compresed_bytes,
        cudaMemcpyDeviceToHost
        ));
    auto data(pressio_data::move(
        pressio_byte_dtype,
        host_buffer,
        {*d_compresed_bytes},
        pressio_data_libc_free_fn,
        nullptr
        ));

    pressio_cuda_check(cudaFreeHost(d_compresed_bytes));
    if(d_temp) {
      pressio_cuda_check(cudaFree(d_temp));
    }
    pressio_cuda_check(cudaFree(temp_space));
    pressio_cuda_check(cudaFree(output_space));
    return data;
  }

  cudaStream_t &stream;
  compat::optional<nvcompCascadedSelectorOpts>& g_opts;
  compat::optional<nvcompCascadedFormatOpts>& g_fopts;
};

class pressio_nvcomp: public libpressio_compressor_plugin {
  pressio_options get_options_impl() const override {
    pressio_options opts;

    set_type(opts, "nvcomp_cascade:mode", pressio_option_charptr_type);

    if(g_fopts) {
      set(opts, "nvcomp_cascade:use_bp", g_fopts->use_bp);
      set(opts, "nvcomp_cascade:num_deltas", g_fopts->num_deltas);
      set(opts, "nvcomp_cascade:num_rles", g_fopts->num_RLEs);
    } else {
      set_type(opts, "nvcomp_cascade:use_bp", pressio_option_int32_type);
      set_type(opts, "nvcomp_cascade:num_deltas", pressio_option_int32_type);
      set_type(opts, "nvcomp_cascade:num_rles", pressio_option_int32_type);
    }

    if(g_opts) {
      set(opts, "nvcomp_cascade:num_samples", g_opts->num_samples);
      set(opts, "nvcomp_cascade:sample_size", g_opts->sample_size);
      set(opts, "nvcomp_cascade:seed", g_opts->seed);
    } else {
      set_type(opts, "nvcomp_cascade:num_samples", pressio_option_uint64_type);
      set_type(opts, "nvcomp_cascade:sample_size", pressio_option_uint64_type);
      set_type(opts, "nvcomp_cascade:seed", pressio_option_uint32_type);
    }

    return opts;
  }
  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    set(opts, "pressio:description", R"(nvcomp

    libpressio bindings for the NVIDIA nvcomp GPU compressors
    )");
    set(opts, "nvcomp_cascade:use_bp", "use bit packing");
    set(opts, "nvcomp_cascade:num_deltas", "number of delta encodings to use");
    set(opts, "nvcomp_cascade:num_rles", "number of run length encodings to use");
    set(opts, "nvcomp_cascade:num_samples", "number of samples to use while configuring automatically");
    set(opts, "nvcomp_cascade:sample_size", "size of samples to use while configuring automatically");
    set(opts, "nvcomp_cascade:seed", "seed to use while generating samples");
    return opts;
  }
  pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "pressio:stability", "experimental");
    return opts;
  }
  int set_options_impl(const pressio_options &options) override {
    std::string mode;
    if(get(options, "nvcomp_cascade:mode", &mode) == pressio_options_key_set) {
      if(mode == "auto") {
        g_opts = compat::nullopt;
        g_fopts = compat::nullopt;
      }
    }


    {
      decltype(nvcompCascadedFormatOpts::num_deltas) ndelta;
      decltype(nvcompCascadedFormatOpts::num_RLEs) nrle;
      decltype(nvcompCascadedFormatOpts::use_bp) usebp;
      auto s1 = get(options, "nvcomp_cascade:num_deltas", &ndelta);
      auto s2 = get(options, "nvcomp_cascade:num_rles", &nrle);
      auto s3 = get(options, "nvcomp_cascade:use_bp", &usebp);
      if(s1 == pressio_options_key_set && s2 == pressio_options_key_set && s3 == pressio_options_key_set) {
        nvcompCascadedFormatOpts fopts;
        fopts.use_bp = usebp;
        fopts.num_deltas = ndelta;
        fopts.num_RLEs = nrle;
        g_fopts = fopts;
      } else if (s1 == pressio_options_key_set || s2 == pressio_options_key_set || s3 == pressio_options_key_set) {
        return set_error(1, "you must set num_rles, num_deltas, and use_bp together or not at all");
      }
    }

    {
      decltype(nvcompCascadedSelectorOpts::seed) seed;
      decltype(nvcompCascadedSelectorOpts::num_samples) num_samples;
      decltype(nvcompCascadedSelectorOpts::sample_size) sample_size;

      auto s1 = get(options, "nvcomp_cascade:seed", &seed);
      auto s2 = get(options, "nvcomp_cascade:num_samples", &num_samples);
      auto s3 = get(options, "nvcomp_cascade:sample_size", &sample_size);
      if(s1 == pressio_options_key_set && s2 == pressio_options_key_set && s3 == pressio_options_key_set) {
        nvcompCascadedSelectorOpts opts;
        opts.sample_size = sample_size;
        opts.num_samples = num_samples;
        opts.seed = seed;
        g_opts = opts;
      } else if (s1 == pressio_options_key_set || s2 == pressio_options_key_set || s3 == pressio_options_key_set) {
        return set_error(2, "you must set seed, num_samples, and sample_size together or not at all");
      }
    }
    return 0;
  }


  int compress_impl(const pressio_data *input, struct pressio_data *output) override {
    try {
      *output = pressio_data_for_each<pressio_data>(*input, compress_impl_typed{stream, g_opts, g_fopts});
      return 0;
    } catch(pressio_cuda_error const& ex) {
      return set_error(6, ex.what());
    } catch(nvcomp::NVCompException const& ex) {
      return set_error(3, ex.what());
    }
  }
  int decompress_impl(const pressio_data *input, struct pressio_data *output) override {
    try {
    nvcomp::CascadedDecompressor decompressor;
    //assume for now the data is on the host

    void* compressed_data;
    size_t compressed_bytes = input->size_in_bytes();
    pressio_cuda_check(cudaMalloc(&compressed_data, compressed_bytes));
    pressio_cuda_check(cudaMemcpy(compressed_data, input->data(), compressed_bytes, cudaMemcpyHostToDevice));

    size_t temp_bytes;
    size_t uncompressed_bytes;

    decompressor.configure(
        compressed_data,
        compressed_bytes,
        &temp_bytes,
        &uncompressed_bytes,
        stream);

    void *temp_space, *uncompressed_output;
    pressio_cuda_check(cudaMalloc(&temp_space, temp_bytes));
    pressio_cuda_check(cudaMalloc(&uncompressed_output, uncompressed_bytes));

    decompressor.decompress_async(
        compressed_data, compressed_bytes, temp_space,
        temp_bytes, uncompressed_output, uncompressed_bytes, stream);

    cudaStreamSynchronize(stream);

    pressio_cuda_check(cudaMemcpy(
        output->data(),
        uncompressed_output,
        uncompressed_bytes,
        cudaMemcpyDeviceToHost));
    } catch(pressio_cuda_error const& ex) {
      return set_error(4, ex.what());
    } catch(nvcomp::NVCompException const& ex) {
      return set_error(5, ex.what());
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
  const char* prefix() const override { return "nvcomp_cascade"; }
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

  cudaStream_t stream = 0;
  compat::optional<nvcompCascadedSelectorOpts> g_opts = compat::nullopt;
  compat::optional<nvcompCascadedFormatOpts> g_fopts = compat::nullopt;
};

static pressio_register pressio_nvcomp_register(
    compressor_plugins(),
    "nvcomp_cascade",
    []{
      return compat::make_unique<pressio_nvcomp>();
    }
    );

