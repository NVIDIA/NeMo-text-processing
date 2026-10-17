# Experimental native transducer runtime

Keep Pynini/OpenFst for grammar construction and export byte-transition tables for native execution.
The C++17 evaluator has no OpenFst dependency and does not construct per-input FSTs.
Its complete English entry point includes Unicode whitespace collapse, bracket preprocessing, classification, tagged-text parsing, field permutations, verbalization, the postprocessing FST, Moses detokenization and punctuation restoration.

This is a development prototype, not a supported backend.
The punctuation options enabled in the reference pipeline are implemented natively.
Compatibility targets the pinned English cached-lookahead reference.
Corpus differential tests and synthetic weighted-graph regressions check exact output; this does not establish compatibility with arbitrary grammars or library versions.
Queue selection includes unweighted LIFO and the reference's cached acyclicity.
The former unweighted counterexample is a passing regression, including cached-property and unused-weighted-branch variants.
Passing finite tests does not prove equivalence on every input.
The current evaluator supports standard tropical weights with byte labels on little-endian hosts; generated symbol labels are rejected.
Resource limits raise errors rather than silently pruning paths.
A runtime handle is mutable and must not be called concurrently.

## Build and test

Run from this directory:

```sh
g++ -std=c++17 -O3 -Wall -Wextra -shared -fPIC byte_transducer.cc -o libtn_byte_transducer.so
g++ -std=c++17 -O3 -Wall -Wextra byte_transducer.cc normalize_stream.cc -o normalize_stream
```

Run the tests from the repository root:

```sh
python -m unittest tools.text_processing_deployment.native_runtime.test_byte_transducer tools.text_processing_deployment.native_runtime.test_text_processing
```

The test needs Pynini for its independent OpenFst reference.
Tests check minimum-cost membership and exact outputs separately, including negative arcs, float rounding at merges, lookahead phase merges, projected-output traversal, unsupported cycles, UTF-8, literal brackets, failed verbalization and repeated calls.

Export a normalizer and benchmark independent written/spoken JSONL references:

```sh
python export_grammars.py --cache-dir CACHE --output-dir GRAMMARS
python benchmark.py --library ./libtn_byte_transducer.so --grammars GRAMMARS --input REFERENCE.jsonl --report REPORT.json --mismatches DIFFERENCES.jsonl
```

Each reference row contains written and spoken string fields.
The JSONL benchmark calls the complete native normalizer but retains Python input/output instrumentation.
The standalone normalize_stream executable reads and writes length-framed UTF-8 records with no Python processing or callbacks.
Invoke it as ./normalize_stream GRAMMARS INPUT.frames OUTPUT.frames; each frame is a four-byte little-endian byte count followed by that many UTF-8 bytes.
The driver fails on malformed input, unsupported text, no accepting path or I/O errors; output before an error is partial.
Source-token decoding, corpus selection and provenance/checkpoint integration are separate from this driver.
Use export_fst from byte_transducer.py to export existing tagger, verbalizer and postprocessor FSTs directly.
NativePipeline loads tagger.bin, verbalizer.bin and postprocessor.bin from a supplied directory.
Its normalize method performs complete English processing; normalize_core exposes the transducer stages for diagnostics.
normalize_batch uses one native call for a list of records, with no Python callbacks during normalization.
Batches use the same framing as the standalone driver, allow at most 64 MiB of input or output, and return no partial result on failure.
Embedded NUL characters can occur in crawl text and are rejected even inside an otherwise readable sentence.
Corpus preparation should reject and count those records before batching; the runtime does not strip characters or silently discard records.
A record failure identifies the zero-based batch index, publishes no partial output and leaves the handle usable for the next call.
ByteTransducer evaluates individual exported graphs and returns both output and path cost.
The C ABI also supports native callers without Python.

## Execution and compatibility

The evaluator builds the reachable input/grammar DAG in reusable flat workspaces and settles predecessors in topological order.
Precomputed next-byte sets prune impossible paths.
Forced literal and epsilon chains become output fragments while retaining merge predecessors and the sequence of nonzero floating-point weight additions.
LIFO execution uses the uncompressed graph because compression can change visitation order.
Mode-specialized loops avoid repeated execution-option checks.
Direct composition starts uncompressed to avoid rebuilding the predominantly unweighted English verbalizer lattices.
Classification reproduces prepared-lookahead arc ordering; postprocessing reproduces output projection and epsilon removal before shortest-path selection.
These transformations retain the original graph; they do not replace the linguistic rules with heuristics.

A single native program should preserve stage boundaries.
Globally minimizing a composed classifier/verbalizer graph would not necessarily reproduce selecting the classifier first and then the first accepted field permutation.
Exact output compatibility with the pinned reference remains required before production use.
Equal-cost alternatives depend on float accumulation and traversal order; a higher-precision accumulator is not a compatible replacement.
Lookahead and output-projection modes reject reachable input-epsilon cycles rather than use an incompatible fallback.
The direct-transducer mode retains a bounded cyclic evaluator; its tests check path cost and membership, not universal tie-order equivalence.
Chain compression accounts for the two possible lookahead copies of a grammar state when preserving merges.

The complete entry point implements the cased, deterministic English reference with both punctuation options enabled, postprocessing enabled and a 729-permutation split limit.
It collapses Unicode whitespace before and after normalization, as the corpus wrapper does.
NeMo's failed-verbalization early return, including escaped brackets and backslashes, is preserved.
This is not a configurable replacement for every Normalizer option.
The audited dependency set is NeMo text processing 1.2.0, Pynini 2.1.6.post1, Python Unicode 15.0.0 and sacremoses 0.2.0.
Changing grammar exports or dependencies requires fresh differential validation.
TNBYTE02 exports retain cached acyclicity rather than recomputing it, since that cache affects reference queue choice.
Re-export legacy TNBYTE01 grammars before direct-mode unweighted execution; missing queue metadata is an error there.
Malformed UTF-8, NUL, unsupported labels and resource-limit failures are explicit errors, not silent fallbacks.

Relevant implementation references:

- [RE2 NFA](https://github.com/google/re2/blob/main/re2/nfa.cc): sparse active states and next-byte filtering.
- [Rust regex-automata](https://docs.rs/regex-automata/latest/regex_automata/): multiple execution strategies, bounded lazy DFA caches and one-pass specialization.
- [Hyperscan](https://github.com/intel/hyperscan) and [VectorScan](https://github.com/VectorCamp/vectorscan): hybrid automata, literal matching and architecture-specific acceleration.
- [PCRE2 JIT](https://www.pcre.org/current/doc/html/pcre2jit.html): compiling repeated matching work into machine code.

These are architectural references, not interchangeable implementations of weighted text transduction.
No code from these engines is copied here.

English punctuation processing uses build-time Unicode property tables from export_text_tables.py; regenerate text_tables.h with --output when deliberately changing the reference Python/sacremoses version.
The checked tables identify their source versions, and LICENSE.sacremoses preserves the MIT notice for derived detokenization code and tables.
ASCII decoding has NEON and SSE2 paths with a scalar fallback; define TN_DISABLE_SIMD to compare outputs and timing.
Compare SIMD and scalar execution on both ARM64 and x86-64 when changing text processing.

This backend is opt-in and does not change Normalizer or Sparrowhawk.
Before enabling a new grammar or dependency version, run differential corpus tests and the repository's applicable NeMo/Sparrowhawk checks.
The reference's prepared-lookahead classifier can choose a different equally weighted output from ordinary Pynini composition; this prototype is not a claim of byte-for-byte equivalence to every Normalizer execution mode.
