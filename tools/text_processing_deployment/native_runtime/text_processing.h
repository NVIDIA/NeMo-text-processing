// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 AND MIT
// English detokenization follows sacremoses; see LICENSE.sacremoses.
// Punctuation restoration follows NeMo's character-slot semantics, including deletions.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#if defined(__aarch64__) && !defined(TN_DISABLE_SIMD)
#include <arm_neon.h>
#elif defined(__SSE2__) && !defined(TN_DISABLE_SIMD)
#include <emmintrin.h>
#endif
#include "text_tables.h"

namespace tn_text {
template <size_t N> bool contains(char32_t c, const Range (&ranges)[N]) {
  auto p =
      std::lower_bound(std::begin(ranges), std::end(ranges), c,
                       [](const Range &r, char32_t v) { return r.last < v; });
  return p != std::end(ranges) && p->first <= c;
}
bool space(char32_t c) {
  if (c < 128)
    return c == ' ' || (c >= 9 && c <= 13) || (c >= 28 && c <= 31);
  return contains(c, whitespace);
}
std::u32string decode(std::string_view s) {
  std::u32string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    // Widen ASCII blocks without a branch or UTF-8 decoder call per character.
#if defined(__aarch64__) && !defined(TN_DISABLE_SIMD)
    if (i + 16 <= s.size()) {
      auto v = vld1q_u8(reinterpret_cast<const uint8_t *>(s.data() + i));
      if (vmaxvq_u8(v) < 128) {
        size_t n = out.size();
        out.resize(n + 16);
        auto lo = vmovl_u8(vget_low_u8(v)), hi = vmovl_u8(vget_high_u8(v));
        auto *p = reinterpret_cast<uint32_t *>(out.data() + n);
        vst1q_u32(p, vmovl_u16(vget_low_u16(lo)));
        vst1q_u32(p + 4, vmovl_u16(vget_high_u16(lo)));
        vst1q_u32(p + 8, vmovl_u16(vget_low_u16(hi)));
        vst1q_u32(p + 12, vmovl_u16(vget_high_u16(hi)));
        i += 16;
        continue;
      }
    }
#elif defined(__SSE2__) && !defined(TN_DISABLE_SIMD)
    if (i + 16 <= s.size()) {
      auto v = _mm_loadu_si128(reinterpret_cast<const __m128i *>(s.data() + i));
      if (!_mm_movemask_epi8(v)) {
        size_t n = out.size();
        out.resize(n + 16);
        auto z = _mm_setzero_si128(), lo = _mm_unpacklo_epi8(v, z),
             hi = _mm_unpackhi_epi8(v, z);
        auto *p = reinterpret_cast<__m128i *>(out.data() + n);
        _mm_storeu_si128(p, _mm_unpacklo_epi16(lo, z));
        _mm_storeu_si128(p + 1, _mm_unpackhi_epi16(lo, z));
        _mm_storeu_si128(p + 2, _mm_unpacklo_epi16(hi, z));
        _mm_storeu_si128(p + 3, _mm_unpackhi_epi16(hi, z));
        i += 16;
        continue;
      }
    }
#endif
    uint32_t c = static_cast<uint8_t>(s[i++]), minimum = 0;
    size_t extra = 0;
    if (c < 128) {
      out.push_back(c);
      continue;
    }
    if (c >= 0xc2 && c <= 0xdf) {
      c &= 31;
      extra = 1;
      minimum = 0x80;
    } else if (c >= 0xe0 && c <= 0xef) {
      c &= 15;
      extra = 2;
      minimum = 0x800;
    } else if (c >= 0xf0 && c <= 0xf4) {
      c &= 7;
      extra = 3;
      minimum = 0x10000;
    } else
      throw std::runtime_error("Invalid UTF-8 lead byte");
    if (i + extra > s.size())
      throw std::runtime_error("Truncated UTF-8");
    for (size_t j = 0; j < extra; ++j) {
      uint8_t b = static_cast<uint8_t>(s[i++]);
      if ((b & 0xc0) != 0x80)
        throw std::runtime_error("Invalid UTF-8 continuation");
      c = (c << 6) | (b & 63);
    }
    if (c < minimum || c > 0x10ffff || (c >= 0xd800 && c <= 0xdfff))
      throw std::runtime_error("Invalid UTF-8 code point");
    out.push_back(c);
  }
  if (out.find(U'\0') != std::u32string::npos)
    throw std::runtime_error("Embedded NUL is unsupported");
  return out;
}
std::string encode(const std::u32string &s) {
  std::string out;
  out.reserve(s.size());
  for (uint32_t c : s) {
    if (c < 0x80)
      out.push_back(c);
    else if (c < 0x800) {
      out.push_back(0xc0 | (c >> 6));
      out.push_back(0x80 | (c & 63));
    } else if (c < 0x10000) {
      out.push_back(0xe0 | (c >> 12));
      out.push_back(0x80 | ((c >> 6) & 63));
      out.push_back(0x80 | (c & 63));
    } else {
      out.push_back(0xf0 | (c >> 18));
      out.push_back(0x80 | ((c >> 12) & 63));
      out.push_back(0x80 | ((c >> 6) & 63));
      out.push_back(0x80 | (c & 63));
    }
  }
  return out;
}
std::u32string collapse(const std::u32string &s) {
  std::u32string out;
  out.reserve(s.size());
  bool pending = false;
  for (char32_t c : s) {
    if (space(c)) {
      pending = !out.empty();
      continue;
    }
    if (pending)
      out += U' ';
    pending = false;
    out += c;
  }
  return out;
}
std::u32string replace_all(const std::u32string &s, std::u32string_view from,
                           std::u32string_view to) {
  std::u32string out;
  out.reserve(s.size());
  size_t p = 0;
  while (p < s.size()) {
    size_t k = s.find(from, p);
    if (k == std::u32string::npos) {
      out.append(s, p, s.size() - p);
      break;
    }
    out.append(s, p, k - p);
    out += to;
    p = k + from.size();
  }
  return out;
}
std::string prepare(const std::u32string &input) {
  std::u32string out;
  out.reserve(input.size() + 8);
  for (char32_t c : input) {
    if (c == U'[' || c == U']') {
      out += U' ';
      out += c;
      out += U' ';
    } else
      out += c;
  }
  return encode(collapse(out));
}
bool all_of(std::u32string_view s, std::u32string_view set) {
  return std::all_of(s.begin(), s.end(),
                     [&](char32_t c) { return set.find(c) != set.npos; });
}
std::u32string detokenize(const std::u32string &input) {
  auto text = replace_all(U" " + input + U" ", U" @-@ ", U"-");
  std::vector<std::u32string_view> tokens;
  for (size_t i = 0; i < text.size();) {
    if (space(text[i])) {
      ++i;
      continue;
    }
    size_t begin = i;
    while (i < text.size() && !space(text[i]))
      ++i;
    tokens.emplace_back(text.data() + begin, i - begin);
  }
  std::u32string out;
  out.reserve(text.size());
  bool prepend = true;
  std::unordered_map<std::u32string, size_t> quotes;
  for (size_t i = 0; i < tokens.size(); ++i) {
    auto token = tokens[i];
    bool currency = std::all_of(token.begin(), token.end(), [](char32_t c) {
      return contains(c, currency_symbols) ||
             std::u32string_view(U"([{¿¡").find(c) != std::u32string_view::npos;
    });
    if (contains(token.front(), cjk)) {
      if ((!i || !contains(tokens[i - 1].back(), cjk)) && prepend)
        out += U' ';
      out += token;
      prepend = true;
    } else if (currency) {
      if (prepend)
        out += U' ';
      out += token;
      prepend = false;
    } else if (all_of(token, U",.?!:;\\%}])") ||
               (i && token.size() > 1 && token[0] == U'\'' &&
                contains(token[1], alpha))) {
      out += token;
      prepend = true;
    } else if (all_of(token, U"'\"„“\x60")) {
      std::u32string key =
          all_of(token, U"„“”") ? U"\"" : std::u32string(token);
      auto &count = quotes[key];
      if (count % 2 == 0) {
        if (token == U"'" && i && tokens[i - 1].back() == U's') {
          out += token;
          prepend = true;
        } else {
          if (prepend)
            out += U' ';
          out += token;
          prepend = false;
          ++count;
        }
      } else {
        out += token;
        prepend = true;
        ++count;
      }
    } else {
      if (prepend)
        out += U' ';
      out += token;
      prepend = true;
    }
  }
  return collapse(out);
}
struct Cell {
  char32_t value;
  bool appended_space = false;
  bool equals(char32_t c) const { return value == c && !appended_space; }
  void append_space() {
    if (!value) {
      value = U' ';
      appended_space = false;
    } else
      appended_space = true;
  }
};
std::u32string restore(std::u32string input, const std::u32string &normalized) {
  if (input.find(U"\x60\x60") != input.npos &&
      normalized.find(U"\x60\x60") == normalized.npos)
    input = replace_all(input, U"\x60\x60", U"\"");
  std::vector<Cell> cells;
  cells.reserve(normalized.size());
  for (auto c : normalized)
    cells.push_back({c, false});
  // Preserve Python's stable character slots: removal does not shift later indices.
  for (char32_t punct :
       std::u32string_view(U"!\"#$%&'()*+,-./:;<=>?@[\\]^_\x60{|}~")) {
    size_t count_in = std::count(input.begin(), input.end(), punct);
    if (!count_in)
      continue;
    size_t count_out =
        std::count_if(cells.begin(), cells.end(),
                      [&](const Cell &c) { return c.equals(punct); });
    bool equal = count_in == count_out;
    size_t j = 0;
    for (size_t i = 0; i < input.size(); ++i) {
      if (input[i] != punct)
        continue;
      while (j < cells.size() && !cells[j].equals(punct))
        ++j;
      if (j == cells.size())
        break;
      bool valid = (i && j && cells[j - 1].equals(input[i - 1])) ||
                   (i + 1 < input.size() && j + 1 < cells.size() &&
                    cells[j + 1].equals(input[i + 1]));
      if (!equal && !valid)
        continue;
      if (i && j) {
        if (cells[j - 1].equals(U' ') && input[i - 1] != U' ')
          cells[j - 1] = {0, false};
        else if (!cells[j - 1].equals(U' ') && input[i - 1] == U' ')
          cells[j - 1].append_space();
      }
      if (i + 1 < input.size() && j + 1 < cells.size()) {
        if (cells[j + 1].equals(U' ') && input[i + 1] != U' ')
          cells[j + 1] = {0, false};
        else if (!cells[j + 1].equals(U' ') && input[i + 1] == U' ')
          cells[j].append_space();
      }
      ++j;
    }
  }
  std::u32string out;
  out.reserve(cells.size());
  for (auto cell : cells) {
    if (cell.value)
      out += cell.value;
    if (cell.appended_space)
      out += U' ';
  }
  return collapse(out);
}
std::string finish(const std::u32string &original, std::string_view output) {
  return encode(restore(original, detokenize(decode(output))));
}
} // namespace tn_text
