// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Experimental byte-transducer evaluator.
// Grammar construction remains in Pynini; this file has no OpenFst dependency.
// State merging follows weighted Viterbi semantics, not regexp leftmost-first semantics.
#include "text_processing.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr uint32_t kNone = UINT32_MAX;
struct Arc {
  uint32_t input, output, target;
  float weight;
};
struct State {
  uint32_t begin, end;
  float final_weight;
};
struct Trace {
  uint32_t parent, output;
};
struct Tail {
  uint32_t target, output;
  std::vector<float> weights;
  std::string input{};
  bool last_epsilon = false;
};
struct Frontier {
  std::vector<float> costs;
  std::vector<uint32_t> paths, active;
  std::vector<uint8_t> queued;
  explicit Frontier(size_t n) : costs(n, kInf), paths(n, kNone), queued(n, 0) {}
  void clear() {
    for (uint32_t s : active) {
      costs[s] = kInf;
      paths[s] = kNone;
      queued[s] = 0;
    }
    active.clear();
  }
};

struct DagNode {
  uint32_t state, pos, begin, end, parent = kNone, via = kNone;
  float cost = kInf;
  uint8_t mark = 0;
};
struct DagEdge {
  uint32_t node, arc;
};
// Reuse per-handle storage and a flat state/position index across records.
struct DagWorkspace {
  std::vector<DagNode> nodes;
  std::vector<DagEdge> edges;
  std::vector<uint32_t> stack, order, slots, used;
  static size_t hash(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    return static_cast<size_t>(key);
  }
  void reset() {
    for (uint32_t i : used)
      slots[i] = kNone;
    used.clear();
    nodes.clear();
    edges.clear();
    stack.clear();
    order.clear();
  }
  void grow() {
    slots.assign(slots.empty() ? 4096 : slots.size() * 2, kNone);
    used.clear();
    for (uint32_t i = 0; i < nodes.size(); ++i) {
      auto &n = nodes[i];
      uint64_t key = (uint64_t(n.pos) << 32) | n.state;
      size_t j = hash(key) & (slots.size() - 1);
      while (slots[j] != kNone)
        j = (j + 1) & (slots.size() - 1);
      slots[j] = i;
      used.push_back(j);
    }
  }
  uint32_t get(uint32_t state, uint32_t pos) {
    if ((nodes.size() + 1) * 2 >= slots.size())
      grow();
    uint64_t key = (uint64_t(pos) << 32) | state;
    size_t j = hash(key) & (slots.size() - 1);
    while (slots[j] != kNone) {
      uint32_t id = slots[j];
      const auto &n = nodes[id];
      if (n.state == state && n.pos == pos)
        return id;
      j = (j + 1) & (slots.size() - 1);
    }
    if (nodes.size() >= 10000000)
      throw std::runtime_error("DAG workspace limit");
    uint32_t id = nodes.size();
    nodes.push_back({state, pos, 0, 0});
    slots[j] = id;
    used.push_back(j);
    return id;
  }
};

class Program {
public:
  std::vector<State> states;
  std::vector<Arc> arcs;
  uint32_t start;
  bool grammar_zero = true, known_acyclic = false, has_queue_properties = false;
  std::vector<std::array<uint64_t, 5>> next_labels;
  std::vector<Tail> tails, literal_tails;
  std::vector<std::string> fragments;
  std::vector<Trace> traces;
  std::vector<uint32_t> queue;
  Frontier first, second;
  std::string result;
  size_t relaxations = 0;

  explicit Program(const char *path) : start(0), first(0), second(0) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
      throw std::runtime_error("Cannot open transducer");
    auto read = [&](void *p, size_t n) {
      if (!f.read(static_cast<char *>(p), n))
        throw std::runtime_error("Truncated transducer");
    };
    const uint32_t endian = 1;
    if (*reinterpret_cast<const uint8_t *>(&endian) != 1)
      throw std::runtime_error("This prototype requires a little-endian host");
    char magic[8];
    read(magic, 8);
    has_queue_properties = std::memcmp(magic, "TNBYTE02", 8) == 0;
    if (!has_queue_properties && std::memcmp(magic, "TNBYTE01", 8))
      throw std::runtime_error("Invalid transducer format");
    uint32_t n, m;
    read(&n, 4);
    read(&m, 4);
    read(&start, 4);
    if (has_queue_properties) {
      uint32_t flags;
      read(&flags, 4);
      if (flags & ~1u)
        throw std::runtime_error("Unsupported queue properties");
      known_acyclic = flags & 1;
    }
    if (!n || start >= n || n > 10000000 || m > 100000000)
      throw std::runtime_error("Invalid transducer dimensions");
    states.resize(n);
    arcs.resize(m);
    for (auto &s : states) {
      read(&s.begin, 4);
      read(&s.end, 4);
      read(&s.final_weight, 4);
      if (s.begin > s.end || s.end > m || std::isnan(s.final_weight) ||
          s.final_weight == -kInf)
        throw std::runtime_error("Invalid state");
    }
    for (auto &a : arcs) {
      read(&a.input, 4);
      read(&a.output, 4);
      read(&a.target, 4);
      read(&a.weight, 4);
      if (a.input > 255 || a.output > 255 || a.target >= n ||
          !std::isfinite(a.weight))
        throw std::runtime_error("Unsupported arc");
    }
    for (const auto &s : states) {
      for (uint32_t i = s.begin + (s.begin < s.end); i < s.end; ++i)
        if (arcs[i - 1].input > arcs[i].input)
          throw std::runtime_error("Unsorted arcs");
    }
    for (const auto &arc : arcs)
      grammar_zero &= arc.weight == 0;
    // Compute possible next bytes through epsilon paths once, rather than
    // exploring paths that cannot consume the current byte on every call.
    next_labels.resize(n);
    std::vector<std::vector<uint32_t>> predecessors(n);
    std::deque<uint32_t> pending;
    std::vector<uint8_t> queued(n, 1);
    for (uint32_t s = 0; s < n; ++s) {
      if (states[s].final_weight != kInf)
        next_labels[s][4] = 1;
      for (uint32_t j = states[s].begin; j < states[s].end; ++j) {
        const auto &a = arcs[j];
        if (!a.input)
          predecessors[a.target].push_back(s);
        else
          next_labels[s][a.input / 64] |= uint64_t{1} << (a.input % 64);
      }
      pending.push_back(s);
    }
    while (!pending.empty()) {
      uint32_t s = pending.front();
      pending.pop_front();
      queued[s] = 0;
      for (uint32_t before : predecessors[s]) {
        bool changed = false;
        for (size_t k = 0; k < 5; ++k) {
          uint64_t combined = next_labels[before][k] | next_labels[s][k];
          changed |= combined != next_labels[before][k];
          next_labels[before][k] = combined;
        }
        if (changed && !queued[before]) {
          pending.push_back(before);
          queued[before] = 1;
        }
      }
    }
    // Fuse forced epsilon chains into output fragments without changing the
    // order of floating-point weight additions.
    // A lookahead state can have both an ordinary and a pending-label copy.
    // Their outgoing arcs can merge even when the grammar target has one predecessor.
    std::vector<uint8_t> phases(n, 0);
    phases[start] = 1;
    for (const auto &a : arcs)
      phases[a.target] |= a.input ? 1 : 2;
    std::vector<uint32_t> indegree(n, 0);
    for (uint32_t source = 0; source < n; ++source)
      for (uint32_t a = states[source].begin; a < states[source].end; ++a)
        indegree[arcs[a].target] += phases[source] == 3 ? 2 : 1;
    auto compile_tails = [&](bool include_input) {
      std::vector<Tail> compiled;
      compiled.reserve(n);
      for (uint32_t s = 0; s < n; ++s) {
        Tail tail{s, 0, {}};
        std::string fragment;
        size_t steps = 0;
        while (steps < 256) {
          const auto &st = states[tail.target];
          // Keep merge points: a preference can disappear if a large later cost
          // is added before the competing prefix costs are compared.
          if (indegree[tail.target] > 1 || st.final_weight != kInf ||
              st.end - st.begin != 1 || indegree[arcs[st.begin].target] > 1 ||
              (!include_input && arcs[st.begin].input))
            break;
          const auto &a = arcs[st.begin];
          if (a.output)
            fragment.push_back(static_cast<char>(a.output));
          if (a.input)
            tail.input.push_back(static_cast<char>(a.input));
          tail.last_epsilon = !a.input;
          if (a.weight != 0)
            tail.weights.push_back(a.weight);
          tail.target = a.target;
          ++steps;
        }
        // Leave long chains and cycles to the bounded general evaluator.
        if (steps == 256) {
          tail = Tail{s, 0, {}};
          fragment.clear();
        }
        if (!fragment.empty()) {
          tail.output = static_cast<uint32_t>(fragments.size() + 256);
          fragments.push_back(std::move(fragment));
        }
        compiled.push_back(std::move(tail));
      }
      return compiled;
    };
    tails = compile_tails(false);
    literal_tails = compile_tails(true);
    first = Frontier(n);
    second = Frontier(n);
    traces.reserve(65536);
    queue.reserve(4096);
  }

  bool relax(Frontier &f, uint32_t s, float cost, uint32_t parent,
             uint32_t output) {
    const auto &tail = tails[s];
    for (float w : tail.weights)
      cost += w;
    s = tail.target;
    if (!(cost < f.costs[s]))
      return false;
    if (++relaxations > 10000000)
      throw std::runtime_error("Relaxation limit reached; possible epsilon "
                               "cycle or excessive ambiguity");
    if (f.costs[s] == kInf)
      f.active.push_back(s);
    if (output) {
      if (traces.size() >= 10000000)
        throw std::runtime_error("Trace memory limit reached");
      traces.push_back({parent, output});
      parent = static_cast<uint32_t>(traces.size() - 1);
    }
    if (tail.output) {
      if (traces.size() >= 10000000)
        throw std::runtime_error("Trace memory limit reached");
      traces.push_back({parent, tail.output});
      parent = static_cast<uint32_t>(traces.size() - 1);
    }
    f.costs[s] = cost;
    f.paths[s] = parent;
    return true;
  }

  bool can_consume(uint32_t s, uint32_t label) const {
    return next_labels[s][label / 64] & (uint64_t{1} << (label % 64));
  }

  void epsilon_closure(Frontier &f, uint32_t label) {
    queue.clear();
    for (uint32_t s : f.active)
      if (can_consume(s, label)) {
        queue.push_back(s);
        f.queued[s] = 1;
      }
    for (size_t p = 0; p < queue.size(); ++p) {
      uint32_t s = queue[p];
      f.queued[s] = 0;
      const float cost = f.costs[s];
      const uint32_t path = f.paths[s];
      const State &st = states[s];
      for (uint32_t j = st.begin; j < st.end && arcs[j].input == 0; ++j) {
        const Arc &a = arcs[j];
        if (can_consume(a.target, label) &&
            relax(f, a.target, cost + a.weight, path, a.output) &&
            !f.queued[tails[a.target].target]) {
          uint32_t terminal = tails[a.target].target;
          queue.push_back(terminal);
          f.queued[terminal] = 1;
        }
      }
    }
  }

  bool project_output = false;
  bool rewrite_projected(float &best_cost, size_t input_size) {
    struct PArc {
      uint32_t target, output;
      float weight;
    };
    const auto &nodes = dag.nodes;
    size_t n = nodes.size();
    std::vector<std::vector<PArc>> outgoing(n);
    std::vector<float> final(n, kInf), distance(n, kInf);
    std::vector<uint8_t> incoming(n, 0), seen(n, 0);
    std::vector<uint32_t> rank(n), pending, closure;
    incoming[0] = 1;
    for (uint32_t i = 0; i < dag.order.size(); ++i)
      rank[dag.order[i]] = i;
    for (uint32_t i = 0; i < n; ++i) {
      if (nodes[i].pos == input_size)
        final[i] = states[nodes[i].state / 2].final_weight;
      for (uint32_t e = nodes[i].begin; e < nodes[i].end; ++e) {
        auto edge = dag.edges[e];
        const auto &arc = arcs[edge.arc];
        outgoing[i].push_back({edge.node, arc.output, arc.weight});
        if (arc.output)
          incoming[edge.node] = 1;
      }
    }
    // NeMo's top_rewrite projects output labels, removes epsilon paths, then
    // selects a path; direct transducer traversal has different tie behavior.
    for (uint32_t source : dag.order) {
      if (!incoming[source])
        continue;
      if (std::none_of(outgoing[source].begin(), outgoing[source].end(),
                       [](const PArc &arc) { return !arc.output; })) {
        std::reverse(outgoing[source].begin(), outgoing[source].end());
        continue;
      }
      closure.clear();
      pending.clear();
      pending.push_back(source);
      while (!pending.empty()) {
        uint32_t id = pending.back();
        pending.pop_back();
        if (seen[id])
          continue;
        seen[id] = 1;
        closure.push_back(id);
        for (const auto &arc : outgoing[id])
          if (!arc.output)
            pending.push_back(arc.target);
      }
      for (uint32_t id : closure) {
        seen[id] = 0;
        distance[id] = kInf;
      }
      std::sort(closure.begin(), closure.end(),
                [&](uint32_t a, uint32_t b) { return rank[a] > rank[b]; });
      distance[source] = 0;
      for (uint32_t id : closure)
        for (const auto &arc : outgoing[id])
          if (!arc.output)
            distance[arc.target] =
                std::min(distance[arc.target], distance[id] + arc.weight);
      std::vector<PArc> expanded;
      std::unordered_map<uint64_t, uint32_t> positions;
      pending.push_back(source);
      float final_weight = kInf;
      while (!pending.empty()) {
        uint32_t id = pending.back();
        pending.pop_back();
        if (seen[id])
          continue;
        seen[id] = 1;
        for (const auto &arc : outgoing[id]) {
          if (!arc.output) {
            if (!seen[arc.target])
              pending.push_back(arc.target);
            continue;
          }
          uint64_t key = (uint64_t(arc.output) << 32) | arc.target;
          auto [it, inserted] = positions.emplace(key, expanded.size());
          float weight = distance[id] + arc.weight;
          if (inserted)
            expanded.push_back({arc.target, arc.output, weight});
          else
            expanded[it->second].weight =
                std::min(expanded[it->second].weight, weight);
        }
        final_weight = std::min(final_weight, distance[id] + final[id]);
      }
      for (uint32_t id : closure)
        seen[id] = 0;
      // OpenFst restores the collected arcs by popping from the back.
      std::reverse(expanded.begin(), expanded.end());
      outgoing[source] = std::move(expanded);
      final[source] = final_weight;
    }
    // Epsilon removal changes arc order, so recompute the reference DFS order.
    std::vector<uint32_t> cursor(n, 0), order;
    pending.push_back(0);
    seen[0] = 1;
    while (!pending.empty()) {
      uint32_t id = pending.back();
      if (cursor[id] < outgoing[id].size()) {
        uint32_t target = outgoing[id][cursor[id]++].target;
        if (!seen[target]) {
          seen[target] = 1;
          pending.push_back(target);
        }
      } else {
        order.push_back(id);
        pending.pop_back();
      }
    }
    std::vector<uint8_t> coaccessible(n, 0);
    bool unweighted = true;
    for (uint32_t id : order) {
      coaccessible[id] = final[id] != kInf;
      for (const auto &arc : outgoing[id]) {
        coaccessible[id] |= coaccessible[arc.target];
        if (coaccessible[arc.target])
          unweighted &= arc.weight == 0;
      }
    }
    // RmEpsilon clears the known acyclic property; ignore dead paths as Connect does.
    if (unweighted) {
      std::fill(seen.begin(), seen.end(), 0);
      pending.clear();
      order.clear();
      pending.push_back(0);
      seen[0] = 1;
      while (!pending.empty()) {
        uint32_t id = pending.back();
        pending.pop_back();
        order.push_back(id);
        for (const auto &arc : outgoing[id])
          if (coaccessible[arc.target] && !seen[arc.target]) {
            seen[arc.target] = 1;
            pending.push_back(arc.target);
          }
      }
      std::reverse(order.begin(), order.end());
    }
    std::fill(distance.begin(), distance.end(), kInf);
    distance[0] = 0;
    std::vector<uint32_t> parent(n, kNone), output(n, 0);
    best_cost = kInf;
    uint32_t last = kNone;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      uint32_t id = *it;
      float cost = distance[id] + final[id];
      if (cost < best_cost) {
        best_cost = cost;
        last = id;
      }
      for (const auto &arc : outgoing[id]) {
        cost = distance[id] + arc.weight;
        if (cost < distance[arc.target]) {
          distance[arc.target] = cost;
          parent[arc.target] = id;
          output[arc.target] = arc.output;
        }
      }
    }
    result.clear();
    if (last == kNone)
      return false;
    for (uint32_t id = last; parent[id] != kNone; id = parent[id])
      result.push_back(output[id]);
    std::reverse(result.begin(), result.end());
    return true;
  }

  DagWorkspace dag;
  bool push_prefix = false;
  // Removed input epsilon arcs still affect AutoQueue and terminal lookahead phase.
  bool input_had_nul = false, trailing_nul = false;
  // Specialize record execution so each arc does not retest the same mode flags.
  template <bool compress, bool lookahead>
  bool rewrite_dag_impl(const char *data, size_t size, float &best_cost) {
    if (size > UINT32_MAX)
      throw std::runtime_error("Input length limit");
    auto &nodes = dag.nodes;
    auto &edges = dag.edges;
    auto &stack = dag.stack;
    auto &order = dag.order;
    dag.reset();
    for (size_t i = 0; i < size; ++i)
      if (!data[i])
        throw std::runtime_error("Embedded NUL is unsupported");
    dag.get(start * 2, 0);
    stack.push_back(0);
    while (!stack.empty()) {
      uint32_t id = stack.back();
      if (!nodes[id].mark) {
        uint32_t state = nodes[id].state / 2, pos = nodes[id].pos;
        uint32_t label =
            pos < size ? static_cast<unsigned char>(data[pos]) : 256;
        nodes[id].mark = 1;
        nodes[id].begin = edges.size();
        const auto &st = states[state];
        auto add = [&](uint32_t a) {
          const auto &arc = arcs[a];
          uint32_t nextpos = pos + (arc.input != 0);
          uint32_t nextlabel =
              nextpos < size ? static_cast<unsigned char>(data[nextpos]) : 256;
          if (!can_consume(arc.target, nextlabel))
            return;
          const auto &tail = literal_tails[arc.target];
          uint32_t terminal = arc.target;
          bool last_epsilon = !arc.input;
          if (compress && tail.target != arc.target) {
            if (tail.input.size() > size - nextpos ||
                std::memcmp(data + nextpos, tail.input.data(),
                            tail.input.size()))
              return;
            nextpos += tail.input.size();
            terminal = tail.target;
            if (terminal != arc.target)
              last_epsilon = tail.last_epsilon;
            nextlabel = nextpos < size
                            ? static_cast<unsigned char>(data[nextpos])
                            : 256;
            if (!can_consume(terminal, nextlabel))
              return;
          }
          bool prefix = lookahead && (nextpos < size || trailing_nul) && last_epsilon;
          uint32_t target = dag.get(terminal * 2 + prefix, nextpos);
          if (edges.size() >= 10000000)
            throw std::runtime_error("DAG edge limit");
          edges.push_back({target, a});
        };
        uint32_t epsilon_end = st.begin;
        while (epsilon_end < st.end && !arcs[epsilon_end].input)
          ++epsilon_end;
        auto add_eps = [&]() {
          for (uint32_t a = st.begin; a < epsilon_end; ++a)
            add(a);
        };
        auto add_match = [&]() {
          if (label == 256)
            return;
          auto it = std::lower_bound(
              arcs.begin() + epsilon_end, arcs.begin() + st.end, label,
              [](const Arc &arc, uint32_t value) { return arc.input < value; });
          for (; it != arcs.begin() + st.end && it->input == label; ++it)
            add(it - arcs.begin());
        };
        // Prepared label lookahead lists the pushed label before epsilon arcs.
        // That order determines the selected output when costs tie.
        if (nodes[id].state & 1) {
          add_match();
          add_eps();
        } else {
          add_eps();
          add_match();
        }
        nodes[id].end = edges.size();
      }
      if (nodes[id].parent == kNone)
        nodes[id].parent = nodes[id].begin;
      if (nodes[id].parent < nodes[id].end) {
        uint32_t child = edges[nodes[id].parent++].node;
        if (nodes[child].mark == 1) {
          if (lookahead || project_output)
            throw std::runtime_error(
                "Input-epsilon cycles are unsupported in compatibility modes");
          return rewrite_fast(data, size, best_cost);
        }
        if (nodes[child].mark == 0)
          stack.push_back(child);
      } else {
        nodes[id].mark = 2;
        nodes[id].parent = kNone;
        order.push_back(id);
        stack.pop_back();
      }
    }
    if (project_output)
      return rewrite_projected(best_cost, size);
    // Only direct composition needs Connect pruning to choose its queue.
    std::vector<uint8_t> coaccessible;
    if (!lookahead && !known_acyclic) {
      coaccessible.assign(nodes.size(), 0);
      for (uint32_t id : order) {
        coaccessible[id] = nodes[id].pos == size &&
                           states[nodes[id].state / 2].final_weight != kInf;
        for (uint32_t e = nodes[id].begin; e < nodes[id].end; ++e)
          coaccessible[id] |= coaccessible[edges[e].node];
      }
    }
    bool unweighted = grammar_zero;
    if (!unweighted && (lookahead || !known_acyclic)) {
      unweighted = true;
      for (const auto &edge : edges) {
        const auto &arc = arcs[edge.arc];
        if (!lookahead && !coaccessible[edge.node])
          continue;
        if (arc.weight != 0 ||
            (compress && !literal_tails[arc.target].weights.empty())) {
          unweighted = false;
          break;
        }
      }
    }
    // AutoQueue tests known acyclicity before falling back to unweighted LIFO.
    // Prepared lookahead retains the acyclic property for empty input only.
    bool lifo = unweighted && !(lookahead ? size == 0 && !input_had_nul : known_acyclic);
    if (lifo && !lookahead && !has_queue_properties)
      throw std::runtime_error(
          "Re-export legacy grammar with cached queue properties");
    if (lifo && compress)
      return rewrite_dag_impl<false, lookahead>(data, size, best_cost);
    if (lifo) {
      std::vector<uint8_t> seen(nodes.size(), 0);
      stack.clear();
      order.clear();
      stack.push_back(0);
      seen[0] = 1;
      while (!stack.empty()) {
        uint32_t id = stack.back();
        stack.pop_back();
        order.push_back(id);
        for (uint32_t e = nodes[id].begin; e < nodes[id].end; ++e) {
          uint32_t target = edges[e].node;
          if (!lookahead && !coaccessible[target])
            continue;
          if (!seen[target]) {
            seen[target] = 1;
            stack.push_back(target);
          }
        }
      }
      std::reverse(order.begin(), order.end());
    }
    // Settle every predecessor before propagating a cost or choosing a tie.
    nodes[0].cost = 0;
    best_cost = kInf;
    uint32_t last = kNone;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
      uint32_t id = *it;
      const auto &node = nodes[id];
      if (node.pos == size) {
        float cost = node.cost + states[node.state / 2].final_weight;
        if (cost < best_cost) {
          best_cost = cost;
          last = id;
        }
      }
      for (uint32_t e = node.begin; e < node.end; ++e) {
        const auto &edge = edges[e];
        const auto &arc = arcs[edge.arc];
        float cost = node.cost + arc.weight;
        if (compress)
          for (float weight : literal_tails[arc.target].weights)
            cost += weight;
        auto &next = nodes[edge.node];
        if (cost < next.cost) {
          next.cost = cost;
          next.parent = id;
          next.via = edge.arc;
        }
      }
    }
    result.clear();
    if (last == kNone)
      return false;
    for (uint32_t id = last; nodes[id].parent != kNone; id = nodes[id].parent) {
      const auto &arc = arcs[nodes[id].via];
      const auto &tail = literal_tails[arc.target];
      if (compress && tail.output) {
        const auto &fragment = fragments[tail.output - 256];
        result.append(fragment.rbegin(), fragment.rend());
      }
      if (arc.output)
        result.push_back(arc.output);
    }
    std::reverse(result.begin(), result.end());
    return true;
  }
  bool rewrite_dag(const char *data, size_t size, float &cost) {
    if (push_prefix) {
      if (!grammar_zero && !project_output)
        return rewrite_dag_impl<true, true>(data, size, cost);
      return rewrite_dag_impl<false, true>(data, size, cost);
    }
    // Verbalizer lattices are usually unweighted; build once without compression.
    return rewrite_dag_impl<false, false>(data, size, cost);
  }
  bool rewrite(const char *data, size_t size, float &cost) {
    // Pynini compiles byte zero as epsilon, not a consuming input label.
    // Keep it in the original text for punctuation and failed-verbalization fallback.
    input_had_nul = std::memchr(data, 0, size) != nullptr;
    trailing_nul = size && !data[size - 1];
    if (input_had_nul) {
      if (!push_prefix && !project_output)
        throw std::runtime_error("NUL requires lookahead or output-projected mode");
      std::string input;
      input.reserve(size);
      for (size_t i = 0; i < size; ++i)
        if (data[i])
          input.push_back(data[i]);
      return rewrite_dag(input.data(), input.size(), cost);
    }
    return rewrite_dag(data, size, cost);
  }

  bool rewrite_fast(const char *data, size_t size, float &best_cost) {
    first.clear();
    second.clear();
    traces.clear();
    result.clear();
    relaxations = 0;
    Frontier *current = &first;
    Frontier *next = &second;
    relax(*current, start, 0, kNone, 0);
    for (size_t p = 0; p < size; ++p) {
      const uint32_t ch = static_cast<unsigned char>(data[p]);
      epsilon_closure(*current, ch);
      // Label zero denotes epsilon, so literal NUL requires a different alphabet.
      if (!ch)
        throw std::runtime_error("Embedded NUL is unsupported");
      next->clear();
      for (uint32_t s : current->active) {
        const State &st = states[s];
        auto a = std::lower_bound(
            arcs.begin() + st.begin, arcs.begin() + st.end, ch,
            [](const Arc &a, uint32_t label) { return a.input < label; });
        for (; a != arcs.begin() + st.end && a->input == ch; ++a)
          relax(*next, a->target, current->costs[s] + a->weight,
                current->paths[s], a->output);
      }
      std::swap(current, next);
      if (current->active.empty()) {
        best_cost = kInf;
        return false;
      }
    }
    epsilon_closure(*current, 256);
    best_cost = kInf;
    uint32_t path = kNone;
    for (uint32_t s : current->active) {
      float cost = current->costs[s] + states[s].final_weight;
      if (cost < best_cost) {
        best_cost = cost;
        path = current->paths[s];
      }
    }
    if (best_cost == kInf)
      return false;
    for (; path != kNone; path = traces[path].parent) {
      uint32_t output = traces[path].output;
      if (output < 256)
        result.push_back(static_cast<char>(output));
      else {
        const auto &fragment = fragments[output - 256];
        result.append(fragment.rbegin(), fragment.rend());
      }
    }
    std::reverse(result.begin(), result.end());
    return true;
  }
};
thread_local std::string error;
} // namespace

extern "C" {
void *tn_load(const char *path) {
  try {
    return new Program(path);
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void tn_delete(void *p) { delete static_cast<Program *>(p); }
const char *tn_error() { return error.c_str(); }
// Result memory belongs to the handle and is valid until its next rewrite.
// One handle is not safe for simultaneous calls.
int tn_rewrite(void *p, const char *text, size_t length, const char **output,
               size_t *size, float *cost) {
  try {
    if (!p || !text || !output || !size || !cost)
      throw std::runtime_error("Null argument");
    auto &program = *static_cast<Program *>(p);
    bool found = program.rewrite(text, length, *cost);
    *output = program.result.data();
    *size = program.result.size();
    return found ? 0 : 1;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
}

namespace {
struct Field {
  enum Kind { String, Boolean, Object, Missing } kind;
  std::string key, value;
  std::vector<Field> children;
};

class TagParser {
  const std::string &text;
  size_t pos = 0;
  void spaces() {
    while (pos < text.size() && text[pos] == ' ')
      ++pos;
  }
  void expect(char c) {
    if (pos == text.size() || text[pos++] != c)
      throw std::runtime_error("Invalid tagged text");
  }

public:
  explicit TagParser(const std::string &input) : text(input) {}
  std::vector<Field> parse(size_t depth = 0) {
    if (depth > 64)
      throw std::runtime_error("Tagged nesting limit exceeded");
    std::vector<Field> result;
    spaces();
    while (pos < text.size() && text[pos] != '}') {
      size_t begin = pos;
      while (pos < text.size() &&
             ((text[pos] >= 'a' && text[pos] <= 'z') ||
              (text[pos] >= 'A' && text[pos] <= 'Z') || text[pos] == '_'))
        ++pos;
      if (pos == begin)
        throw std::runtime_error("Invalid tag key");
      Field f;
      f.key = text.substr(begin, pos - begin);
      spaces();
      if (f.key == "preserve_order") {
        expect(':');
        spaces();
        if (text.compare(pos, 4, "true") != 0)
          throw std::runtime_error("Invalid preserve_order");
        pos += 4;
        f.kind = Field::Boolean;
      } else if (pos < text.size() && text[pos] == '{') {
        ++pos;
        f.kind = Field::Object;
        f.children = parse(depth + 1);
        expect('}');
      } else {
        expect(':');
        spaces();
        expect('"');
        begin = pos;
        while (pos + 1 < text.size() &&
               !(text[pos] == '"' && text[pos + 1] == ' '))
          ++pos;
        if (pos + 1 >= text.size())
          throw std::runtime_error("Unterminated tag string");
        f.kind = pos == begin ? Field::Missing : Field::String;
        f.value = text.substr(begin, pos - begin);
        ++pos;
      }
      // NeMo's top-level list preserves repeated "tokens"; nested dictionaries overwrite duplicate keys.
      auto old =
          depth ? std::find_if(result.begin(), result.end(),
                               [&](const Field &x) { return x.key == f.key; })
                : result.end();
      if (old != result.end())
        *old = std::move(f);
      else
        result.push_back(std::move(f));
      spaces();
    }
    if (!depth && pos != text.size())
      throw std::runtime_error("Trailing tag text");
    return result;
  }
};

size_t permutation_count(const std::vector<Field> &fields) {
  size_t result = 1;
  for (size_t i = 2; i <= fields.size(); ++i) {
    if (result > 729 / i)
      return 730;
    result *= i;
  }
  for (const auto &f : fields)
    if (f.kind == Field::Object) {
      size_t sub = permutation_count(f.children);
      if (result > 729 / sub)
        return 730;
      result *= sub;
    }
  return result;
}

std::vector<std::string> permutations(const std::vector<Field> &fields) {
  bool preserve = false;
  std::vector<size_t> order;
  std::vector<std::vector<std::string>> pieces;
  for (size_t i = 0; i < fields.size(); ++i) {
    const auto &f = fields[i];
    order.push_back(i);
    preserve |= f.key == "preserve_order";
    if (f.kind == Field::Missing)
      return {};
    if (f.kind == Field::String)
      pieces.push_back({f.key + ": \"" + f.value + "\" "});
    else if (f.kind == Field::Boolean)
      pieces.push_back({f.key + ": true "});
    else {
      auto values = permutations(f.children);
      for (auto &v : values)
        v = " " + f.key + " { " + v + " } ";
      pieces.push_back(std::move(values));
    }
  }
  std::vector<std::string> result;
  do {
    std::vector<std::string> partial(1);
    for (size_t i : order) {
      std::vector<std::string> expanded;
      for (const auto &before : partial)
        for (const auto &after : pieces[i])
          expanded.push_back(before + after);
      partial = std::move(expanded);
    }
    result.insert(result.end(), partial.begin(), partial.end());
    if (result.size() > 729)
      throw std::runtime_error("Permutation limit exceeded");
  } while (!preserve && std::next_permutation(order.begin(), order.end()));
  return result;
}

std::string collapse_spaces(const std::string &text) {
  std::string result;
  result.reserve(text.size());
  for (char c : text) {
    if (c != ' ' || (!result.empty() && result.back() != ' '))
      result.push_back(c);
  }
  if (!result.empty() && result.back() == ' ')
    result.pop_back();
  return result;
}

class Pipeline {
public:
  Program tagger, verbalizer, postprocessor;
  std::string result, batch_result;
  bool verbalization_failed = false;
  Pipeline(const char *tagger_path, const char *verbalizer_path,
           const char *post_path)
      : tagger(tagger_path), verbalizer(verbalizer_path),
        postprocessor(post_path) {
    tagger.push_prefix = true;
    postprocessor.project_output = true;
  }

  bool rewrite(const char *text, size_t length) {
    float cost;
    result.clear();
    verbalization_failed = false;
    if (!tagger.rewrite(text, length, cost))
      return false;
    auto tokens = TagParser(tagger.result).parse();
    // NeMo attempts verbalization even when the tag parser returns no tokens.
    size_t begin = 0;
    do {
      size_t end = begin, count = 1;
      while (end < tokens.size()) {
        size_t n = permutation_count({tokens[end]});
        if (n > 729)
          throw std::runtime_error("Unsplittable tag permutation group");
        if (count > 729 / n)
          break;
        count *= n;
        ++end;
      }
      std::vector<std::string> candidates(1);
      for (size_t i = begin; i < end; ++i) {
        auto choices = permutations({tokens[i]});
        std::vector<std::string> expanded;
        for (const auto &prefix : candidates)
          for (const auto &choice : choices)
            expanded.push_back(prefix + choice);
        candidates = std::move(expanded);
      }
      bool found = false;
      for (const auto &candidate : candidates) {
        if (verbalizer.rewrite(candidate.data(), candidate.size(), cost)) {
          if (!result.empty())
            result += ' ';
          result += verbalizer.result;
          found = true;
          break;
        }
      }
      if (!found) {
        verbalization_failed = true;
        return false;
      }
      begin = end;
    } while (begin < tokens.size());
    result = collapse_spaces(result);
    if (!result.empty()) {
      if (!postprocessor.rewrite(result.data(), result.size(), cost))
        return false;
      result = postprocessor.result;
    }
    return true;
  }
};
} // namespace
extern "C" {
void *tn_pipeline_load(const char *t, const char *v, const char *p) {
  try {
    return new Pipeline(t, v, p);
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
void tn_pipeline_delete(void *p) { delete static_cast<Pipeline *>(p); }
// Input whitespace/punctuation preprocessing and optional Moses postprocessing
// are outside this experimental core and must be supplied by the caller.
int tn_pipeline_rewrite(void *p, const char *text, size_t length,
                        const char **out, size_t *size) {
  try {
    if (!p || !text || !out || !size)
      throw std::runtime_error("Null argument");
    auto &pipeline = *static_cast<Pipeline *>(p);
    bool found = pipeline.rewrite(text, length);
    *out = pipeline.result.data();
    *size = pipeline.result.size();
    return found ? 0 : 1;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
}

extern "C" {
int tn_pipeline_normalize(void *p, const char *text, size_t length,
                          const char **out, size_t *size) {
  try {
    if (!p || !text || !out || !size)
      throw std::runtime_error("Null argument");
    auto &pipeline = *static_cast<Pipeline *>(p);
    auto original =
        tn_text::collapse(tn_text::decode(std::string_view(text, length)));
    bool found = true;
    if (original.empty())
      pipeline.result.clear();
    else {
      auto prepared = tn_text::prepare(original);
      found = pipeline.rewrite(prepared.data(), prepared.size());
      if (found)
        pipeline.result = tn_text::finish(original, pipeline.result);
      else if (pipeline.verbalization_failed) {
        // NeMo returns its escaped prepared input on failed verbalization.
        // It skips punctuation postprocessing on this early-return path.
        pipeline.result.clear();
        for (char c : prepared) {
          if (c == '[' || c == ']' || c == '\\')
            pipeline.result += '\\';
          pipeline.result += c;
        }
        found = true;
      }
    }
    *out = pipeline.result.data();
    *size = pipeline.result.size();
    return found ? 0 : 1;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
// Consume a bounded batch without Python callbacks between records.
// No partial output is published when any record or frame fails.
int tn_pipeline_normalize_batch(void *p, const char *frames, size_t length,
                                const char **out, size_t *size) {
  if (out)
    *out = nullptr;
  if (size)
    *size = 0;
  try {
    if (!p || !frames || !out || !size)
      throw std::runtime_error("Null argument");
    if (length > 64 * 1024 * 1024)
      throw std::runtime_error("Batch exceeds 64 MiB");
    auto &batch = static_cast<Pipeline *>(p)->batch_result;
    batch.clear();
    size_t position = 0, record = 0;
    while (position < length) {
      if (length - position < 4)
        throw std::runtime_error("Truncated batch frame header");
      const auto *h =
          reinterpret_cast<const unsigned char *>(frames + position);
      uint32_t n = uint32_t(h[0]) | (uint32_t(h[1]) << 8) |
                   (uint32_t(h[2]) << 16) | (uint32_t(h[3]) << 24);
      position += 4;
      if (n > 16 * 1024 * 1024 || n > length - position)
        throw std::runtime_error("Invalid batch frame length");
      const char *value = nullptr;
      size_t value_size = 0;
      int status =
          tn_pipeline_normalize(p, frames + position, n, &value, &value_size);
      if (status)
        throw std::runtime_error("Batch record " + std::to_string(record) +
                                 ": " +
                                 (status < 0 ? error : "No accepting path"));
      if (value_size > UINT32_MAX ||
          value_size + 4 > 64 * 1024 * 1024 - batch.size())
        throw std::runtime_error("Batch output exceeds 64 MiB");
      for (size_t i = 0; i < 4; ++i)
        batch.push_back(static_cast<char>((value_size >> (8 * i)) & 255));
      batch.append(value, value_size);
      position += n;
      ++record;
    }
    *out = batch.data();
    *size = batch.size();
    return 0;
  } catch (const std::exception &e) {
    if (p)
      static_cast<Pipeline *>(p)->batch_result.clear();
    error = e.what();
    return -1;
  }
}
// Independent stage entry points let differential tests isolate text processing
// from weighted-path selection; result memory lasts until the next stage call.
int tn_text_stage(int stage, const char *input, size_t n, const char *output,
                  size_t m, const char **result, size_t *length) {
  try {
    if (!input || !output || !result || !length)
      throw std::runtime_error("Null argument");
    static thread_local std::string value;
    auto original =
        tn_text::collapse(tn_text::decode(std::string_view(input, n)));
    if (stage == 0)
      value = tn_text::prepare(original);
    else if (stage == 1)
      value = tn_text::encode(
          tn_text::detokenize(tn_text::decode(std::string_view(output, m))));
    else if (stage == 2)
      value = tn_text::finish(original, std::string_view(output, m));
    else
      throw std::runtime_error("Invalid text stage");
    *result = value.data();
    *length = value.size();
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
}

extern "C" int tn_set_mode(void *handle, int mode) {
  if (!handle || mode < 0 || mode > 2) {
    error = "Invalid evaluator mode";
    return -1;
  }
  auto &program = *static_cast<Program *>(handle);
  program.push_prefix = mode == 1;
  program.project_output = mode == 2;
  return 0;
}
