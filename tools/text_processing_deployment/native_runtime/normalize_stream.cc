// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Binary framing keeps embedded newlines intact and amortizes process startup.
// Each record is a little-endian uint32 byte length followed by UTF-8 bytes.
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
extern "C" {
void *tn_pipeline_load(const char *, const char *, const char *);
void tn_pipeline_delete(void *);
const char *tn_error();
int tn_pipeline_normalize(void *, const char *, size_t, const char **,
                          size_t *);
}
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::runtime_error("Usage: normalize_stream GRAMMAR_DIRECTORY "
                               "INPUT.frames OUTPUT.frames");
    std::string root = argv[1];
    std::unique_ptr<void, decltype(&tn_pipeline_delete)> pipeline(
        tn_pipeline_load((root + "/tagger.bin").c_str(),
                         (root + "/verbalizer.bin").c_str(),
                         (root + "/postprocessor.bin").c_str()),
        tn_pipeline_delete);
    if (!pipeline)
      throw std::runtime_error(tn_error());
    std::ifstream in(argv[2], std::ios::binary);
    std::ofstream out(argv[3], std::ios::binary);
    if (!in || !out)
      throw std::runtime_error("Cannot open input/output");
    auto start = std::chrono::steady_clock::now();
    size_t count = 0;
    std::string input;
    for (;;) {
      unsigned char h[4];
      in.read(reinterpret_cast<char *>(h), 4);
      if (in.gcount() == 0 && in.eof())
        break;
      if (in.gcount() != 4)
        throw std::runtime_error("Truncated input frame header");
      uint32_t n = uint32_t(h[0]) | (uint32_t(h[1]) << 8) |
                   (uint32_t(h[2]) << 16) | (uint32_t(h[3]) << 24);
      if (n > 16 * 1024 * 1024)
        throw std::runtime_error("Input frame exceeds 16 MiB");
      input.resize(n);
      if (!in.read(input.data(), n))
        throw std::runtime_error("Truncated input frame");
      const char *result = nullptr;
      size_t length = 0;
      int status = tn_pipeline_normalize(pipeline.get(), input.data(),
                                         input.size(), &result, &length);
      if (status)
        throw std::runtime_error(status < 0 ? tn_error() : "No accepting path");
      if (length > UINT32_MAX)
        throw std::runtime_error("Output frame too long");
      for (size_t i = 0; i < 4; ++i)
        h[i] = (length >> (8 * i)) & 255;
      out.write(reinterpret_cast<const char *>(h), 4);
      out.write(result, length);
      if (!out)
        throw std::runtime_error("Output write failed");
      ++count;
    }
    out.flush();
    if (!out)
      throw std::runtime_error("Output flush failed");
    std::cerr << "{\"records\":" << count << ",\"processing_seconds\":"
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << "}\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
