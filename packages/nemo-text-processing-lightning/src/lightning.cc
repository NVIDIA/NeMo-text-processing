// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// nemo-text-processing-lightning — a minimal OpenFst runtime for applying prebuilt grammars.
//
// Scope is the tagger path only.  Read an FST out of a FAR, prepare the
// lookahead artifact once (Invert, then Convert to `olabel_lookahead`), and
// apply it per request with the GIL released.
//
// Three facts drive the shape of this file:
//
//  1. Composition consults lookahead data through its *left* operand only, and
//     the tagger naturally sits on the right, so we compose the inverses and
//     invert the result: A o B = (B^-1 o A^-1)^-1.
//  2. The `olabel_lookahead` conversion renumbers the labels it indexes
//     (reachable sets are stored as intervals, so labels reachable together must
//     be numbered adjacently).  The other operand has to be relabelled
//     identically or composition matches the wrong pairs.  The map is read back
//     off the prepared FST with LabelLookAheadRelabeler::RelabelPairs, which is
//     the same call `fstconvert --save_relabel_opairs` makes; it never crosses
//     into Python.
//  3. Nothing here *references* a lookahead symbol -- MatcherFst is a template
//     and the plugin registrations are pure static-init side effects.  We use
//     the concrete type StdOLabelLookAheadFst rather than the registry, so no
//     dlopen plugin and no libfstlookahead is required at all.  See build.sh.

#ifndef NEMO_TPL_OPENFST_VERSION
#define NEMO_TPL_OPENFST_VERSION "unknown"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include <fst/arc.h>
#include <fst/compose.h>
#include <fst/const-fst.h>
#include <fst/fst.h>
#include <fst/invert.h>
#include <fst/matcher-fst.h>
#include <fst/register.h>
#include <fst/shortest-path.h>
#include <fst/util.h>
#include <fst/vector-fst.h>
#include <fst/extensions/far/far.h>

namespace py = pybind11;

namespace {

using StdArc = fst::StdArc;
using Label = StdArc::Label;
using Weight = StdArc::Weight;
using LaFst = fst::StdOLabelLookAheadFst;
using Relabeler = fst::LabelLookAheadRelabeler<StdArc>;

// Bumped whenever anything about the prepared artifact changes (the conversion,
// the arc type, the OpenFst FST binary format).  Part of the cache key, so a
// stale artifact from an older nemo-text-processing-lightning is never reused.
constexpr int kArtifactVersion = 1;

// The FST type registry is a function-local static inside a template, so with
// -fvisibility=hidden this extension gets its own copy rather than sharing
// libfst's.  That is deliberate -- it is half of why two OpenFst copies can sit
// in one process without binding across -- but it means the registry starts
// empty and FarReader, which reads entries through the generic Fst::Read, has
// nothing to dispatch "vector" to.  So we populate our own.
//
// Registering the lookahead type here is also what makes the *registry* route
// work (fstconvert-style, by type name).  This file does not use it -- it names
// StdOLabelLookAheadFst directly -- which is precisely why no libfstlookahead
// and no dlopen plugin is needed.  See the linker note in build.sh.
static fst::FstRegisterer<fst::VectorFst<StdArc>> vector_registerer;
static fst::FstRegisterer<fst::ConstFst<StdArc>> const_registerer;
static fst::FstRegisterer<LaFst> olabel_lookahead_registerer;

// ---------------------------------------------------------------------------
// cache key

// Non-cryptographic 64-bit content hash (FNV-style mix over 8-byte words).  The
// job is cache invalidation, not integrity: a regenerated grammar must not hit
// a stale artifact.  ~1 GB/s, so hashing the 7.5 MB English tagger costs less
// than reading it.
uint64_t HashFile(const std::string &path) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    throw std::runtime_error("nemo_text_processing_lightning: cannot open " + path);
  }
  uint64_t h = 0xcbf29ce484222325ULL;
  constexpr uint64_t kPrime = 0x100000001b3ULL;
  std::vector<unsigned char> buf(1 << 16);
  uint64_t total = 0;
  size_t n;
  while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
    total += n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      uint64_t w;
      std::memcpy(&w, buf.data() + i, 8);
      h = (h ^ w) * kPrime;
      h ^= h >> 29;
    }
    for (; i < n; ++i) {
      h = (h ^ buf[i]) * kPrime;
    }
  }
  std::fclose(f);
  h = (h ^ total) * kPrime;
  h ^= h >> 31;
  return h;
}

std::string HexKey(uint64_t h, const std::string &key) {
  // Fold the FAR key in so two keys out of one archive cannot share a file.
  uint64_t k = 0x9e3779b97f4a7c15ULL;
  for (unsigned char c : key) k = (k ^ c) * 0x100000001b3ULL;
  char out[64];
  std::snprintf(out, sizeof(out), "v%d-%016llx%016llx", kArtifactVersion,
                static_cast<unsigned long long>(h),
                static_cast<unsigned long long>(k));
  return std::string(out);
}

// ---------------------------------------------------------------------------
// preparation

// Reads one FST out of a FAR archive.  The archives NeMo caches hold a single
// entry under a documented key ("tokenize_and_classify", "verbalize", ...); an
// empty key takes the first entry.
std::unique_ptr<fst::VectorFst<StdArc>> ReadFarFst(const std::string &far_path,
                                                   const std::string &key) {
  std::unique_ptr<fst::FarReader<StdArc>> reader(
      fst::FarReader<StdArc>::Open(far_path));
  if (reader == nullptr) {
    throw std::runtime_error("nemo_text_processing_lightning: cannot open FAR " + far_path);
  }
  if (!key.empty() && !reader->Find(key)) {
    throw std::runtime_error("nemo_text_processing_lightning: key '" + key + "' not in FAR " +
                             far_path);
  }
  if (reader->Done()) {
    throw std::runtime_error("nemo_text_processing_lightning: FAR is empty: " + far_path);
  }
  const fst::Fst<StdArc> *fst = reader->GetFst();
  if (fst == nullptr) {
    throw std::runtime_error("nemo_text_processing_lightning: cannot read FST from " + far_path);
  }
  auto out = std::make_unique<fst::VectorFst<StdArc>>(*fst);
  if (out->Properties(fst::kError, false) & fst::kError) {
    throw std::runtime_error("nemo_text_processing_lightning: FST in " + far_path + " is in error");
  }
  return out;
}

// The label renumbering the conversion applied, in the same (old, new) form
// and with the same `avoid_collisions` extra pairs that
// `fstconvert --save_relabel_opairs` writes.
//
// This can only be asked of a *freshly converted* FST.  The reachability
// interval sets survive Write/Read, but `label2index_` does not -- a reloaded
// artifact answers "LabelReachableData: No relabeling data" and hands back an
// empty map, which is an identity relabelling, which composes to nothing and
// raises no error.  So the map is written beside the artifact and read back
// with it.
std::vector<std::pair<Label, Label>> ExtractRelabelPairs(const LaFst &la) {
  std::vector<std::pair<Label, Label>> pairs;
  Relabeler::RelabelPairs(la, &pairs, /*avoid_collisions=*/true);
  return pairs;
}

bool FileExists(const std::string &path) {
  struct stat st;
  return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string RelabelPathFor(const std::string &artifact) {
  return artifact + ".relabel";
}

// Invert, convert to olabel_lookahead, write atomically.  0.09 s + 0.26 s on the
// 161k-state English tagger, once per grammar build.
std::unique_ptr<LaFst> PrepareArtifact(const std::string &far_path,
                                       const std::string &key,
                                       const std::string &out_path,
                                       std::vector<std::pair<Label, Label>> *pairs) {
  auto plain = ReadFarFst(far_path, key);
  // Composition asks only its *left* operand for lookahead data, and the tagger
  // naturally sits on the right.  Compose the inverses instead; tag() inverts
  // the lattice back.
  fst::Invert(plain.get());
  // The MatcherFst constructor runs the LabelLookAheadRelabeler initialiser,
  // which is where the label renumbering happens.
  auto la = std::make_unique<LaFst>(*plain);
  if (la->Properties(fst::kError, false) & fst::kError) {
    throw std::runtime_error("nemo_text_processing_lightning: olabel_lookahead conversion failed");
  }
  *pairs = ExtractRelabelPairs(*la);
  if (pairs->empty()) {
    throw std::runtime_error("nemo_text_processing_lightning: conversion produced no relabelling map");
  }
  if (!out_path.empty()) {
    // Per-process temporaries, so two workers preparing the same grammar at
    // once cannot scribble on each other; the rename is what publishes.
    const std::string suffix = ".tmp." + std::to_string(getpid());
    const std::string tmp = out_path + suffix;
    const std::string tmp_map = RelabelPathFor(out_path) + suffix;
    if (!la->Write(tmp)) {
      throw std::runtime_error("nemo_text_processing_lightning: cannot write " + tmp);
    }
    if (!fst::WriteLabelPairs(tmp_map, *pairs)) {
      std::remove(tmp.c_str());
      throw std::runtime_error("nemo_text_processing_lightning: cannot write " + tmp_map);
    }
    // Map first, then the FST: the FST's presence is what the cache lookup
    // tests, so it must never appear without its map.
    if (std::rename(tmp_map.c_str(), RelabelPathFor(out_path).c_str()) != 0 ||
        std::rename(tmp.c_str(), out_path.c_str()) != 0) {
      std::remove(tmp.c_str());
      std::remove(tmp_map.c_str());
      throw std::runtime_error("nemo_text_processing_lightning: cannot rename into " + out_path);
    }
  }
  return la;
}

// ---------------------------------------------------------------------------

std::string PathString(const fst::Fst<StdArc> &fst) {
  std::string out;
  auto state = fst.Start();
  if (state == fst::kNoStateId) return out;
  while (true) {
    fst::ArcIterator<fst::Fst<StdArc>> aiter(fst, state);
    if (aiter.Done()) break;
    const StdArc &arc = aiter.Value();
    if (arc.olabel != 0) out.push_back(static_cast<char>(arc.olabel & 0xff));
    state = arc.nextstate;
  }
  return out;
}

class Tagger {
 public:
  Tagger(std::unique_ptr<LaFst> fst,
         const std::vector<std::pair<Label, Label>> &pairs,
         std::string artifact_path, bool prepared)
      : fst_(std::move(fst)),
        artifact_path_(std::move(artifact_path)),
        prepared_(prepared) {
    relabel_.resize(256);
    for (int i = 0; i < 256; ++i) relabel_[i] = i;
    for (const auto &p : pairs) {
      pairs_.emplace(p.first, p.second);
      if (p.first >= 0 && p.first < 256) relabel_[p.first] = p.second;
    }
  }

  std::string Tag(const std::string &text) const {
    // One state per byte.  Only the matched (input) side is relabelled; the
    // output side stays the raw byte so the tagged string reads back unchanged.
    fst::VectorFst<StdArc> acceptor;
    acceptor.ReserveStates(text.size() + 1);
    auto state = acceptor.AddState();
    acceptor.SetStart(state);
    for (unsigned char c : text) {
      auto next = acceptor.AddState();
      acceptor.AddArc(state, StdArc(relabel_[c], c, Weight::One(), next));
      state = next;
    }
    acceptor.SetFinal(state, Weight::One());

    std::string out;
    {
      // The whole point of the package: a 498 ms composition that does not
      // hold the interpreter lock.
      py::gil_scoped_release unlock;
      fst::VectorFst<StdArc> lattice;
      fst::Compose(static_cast<const fst::Fst<StdArc> &>(*fst_), acceptor,
                   &lattice);
      fst::Invert(&lattice);
      fst::VectorFst<StdArc> best;
      // nshortest=1, unique=true, mirroring pynini.shortestpath as the pipeline
      // calls it.
      fst::ShortestPath(lattice, &best, /*nshortest=*/1, /*unique=*/true);
      out = PathString(best);
    }
    return out;
  }

  int64_t NumStates() const { return fst_->NumStates(); }
  const std::string &ArtifactPath() const { return artifact_path_; }
  bool Prepared() const { return prepared_; }
  const std::map<Label, Label> &RelabelPairs() const { return pairs_; }

 private:
  std::unique_ptr<LaFst> fst_;
  std::vector<Label> relabel_;
  std::map<Label, Label> pairs_;
  std::string artifact_path_;
  bool prepared_;
};

std::shared_ptr<Tagger> TaggerFromFar(const std::string &far_path,
                                      const std::string &key,
                                      const std::string &cache_dir) {
  std::string artifact;
  std::unique_ptr<LaFst> la;
  std::vector<std::pair<Label, Label>> pairs;
  bool prepared = false;
  {
    py::gil_scoped_release unlock;
    if (!cache_dir.empty()) {
      // Basename of the FAR, so several grammars can share one cache dir; the
      // hash of its contents, so a regenerated grammar cannot hit a stale
      // artifact.
      std::string stem = far_path;
      const auto slash = stem.find_last_of('/');
      if (slash != std::string::npos) stem = stem.substr(slash + 1);
      artifact = cache_dir + "/" + stem + "." + HexKey(HashFile(far_path), key) +
                 ".la.fst";
      const std::string map_path = RelabelPathFor(artifact);
      if (FileExists(artifact) && FileExists(map_path)) {
        la.reset(LaFst::Read(artifact));
        if (la != nullptr && (la->Properties(fst::kError, false) & fst::kError)) {
          la.reset();  // corrupt, or written by an incompatible OpenFst
        }
        if (la != nullptr && (!fst::ReadLabelPairs(map_path, &pairs) ||
                              pairs.empty())) {
          la.reset();
          pairs.clear();
        }
      }
    }
    if (la == nullptr) {
      la = PrepareArtifact(far_path, key, artifact, &pairs);
      prepared = true;
    }
  }
  return std::make_shared<Tagger>(std::move(la), pairs, artifact, prepared);
}

// Runtime capability check.  Builds a two-state acceptor, converts it, and
// composes through it -- so this returns true only if the lookahead matcher
// actually runs, not merely if the header was in the include path.
bool HasLookahead() {
  try {
    fst::VectorFst<StdArc> f;
    auto s0 = f.AddState();
    auto s1 = f.AddState();
    f.SetStart(s0);
    f.AddArc(s0, StdArc(1, 1, Weight::One(), s1));
    f.SetFinal(s1, Weight::One());
    LaFst la(f);
    if (la.Properties(fst::kError, false) & fst::kError) return false;
    if (la.Type() != "olabel_lookahead") return false;
    if (ExtractRelabelPairs(la).empty()) return false;
    fst::VectorFst<StdArc> out;
    fst::Compose(static_cast<const fst::Fst<StdArc> &>(la), f, &out);
    return !(out.Properties(fst::kError, false) & fst::kError) &&
           out.NumStates() > 0;
  } catch (...) {
    return false;
  }
}

}  // namespace

PYBIND11_MODULE(_lightning, m) {
  // pywrapfst does this too.  Without it an OpenFst error is LOG(FATAL) and
  // takes the interpreter with it instead of raising.
  FST_FLAGS_fst_error_fatal = false;

  m.doc() = "Minimal OpenFst runtime: applies prebuilt lookahead grammars.";
  m.attr("__openfst_version__") = NEMO_TPL_OPENFST_VERSION;

  m.def("has_lookahead", &HasLookahead,
        "True if the olabel_lookahead FST type is usable in this process.");

  py::class_<Tagger, std::shared_ptr<Tagger>>(m, "Tagger")
      .def_static("from_far", &TaggerFromFar, py::arg("far_path"),
                  py::arg("key") = "tokenize_and_classify",
                  py::arg("cache_dir") = "")
      .def("tag", &Tagger::Tag, py::arg("text"))
      .def_property_readonly("num_states", &Tagger::NumStates)
      .def_property_readonly("artifact_path", &Tagger::ArtifactPath)
      .def_property_readonly("prepared", &Tagger::Prepared)
      .def_property_readonly("relabel_pairs", &Tagger::RelabelPairs);
}
