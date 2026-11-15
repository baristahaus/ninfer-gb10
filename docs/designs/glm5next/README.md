# GLM5Next design and implementation index

This directory holds the designs and execution checklists that a concrete implementation must take
over. The long-term references are maintained separately:

- [Model semantics](../../maintainer/glm5next-model.md): complete mathematics, logical weights,
  causal relationships, Main/MTP/Vision and historical dependencies.
- [Quantization and numerical representation](../../maintainer/glm5next-quantization.md): source
  representation, selected quantization, exact encoding, compute precision, explicit rounding and
  weight capacity.

Read the matching long-term reference before you read an implementation design. The interface names
and code fragments here are contracts that still have to land. Do not claim that a runtime
capability exists because of them. Confirmed model and numerical decisions do not reopen because of
implementation progress or a document move.

## 1. Documents and status

The table below records the local state. Completion evidence for a base execution task lives only in
its checklist entry. An unchecked item does not mean that the current implementation lacks support.
Do not infer remote execution progress from a stale local record.

| Content | Decision status | Implementation status / entry |
|---|---|---|
| Architecture, source representation, quantization and numerical policy | Decided | Long-term reference; the new quantization product is not generated yet and no model accuracy evaluation has run |
| [Dual-Spark split and resources](parallel.md) | Module split, owners and mHC replication decided | Communication and model execution integration pending; capacity is derived, and no GLM two-host end-to-end measurement exists yet |
| [Base Op checklist](op-checklist.md) | The existing scope is fixed | Keep the status, verification and performance evidence of the original 20 entries |
| [KDA ReplaySSM](replayssm-kda.md) | Internal gate, recurrent decode/verify/fold and per-round immediate fold decided | Design waits for implementation and acceptance; issued separately |
| [Kpool Sparse MLA](kpool-sparse-mla.md) | Interface, page, codec, selection/tail decided | Design waits for implementation and acceptance; issued separately |
| [mHC](mhc.md) | Five entries, layout, precision, fusion and replicated placement decided | Design waits for implementation and acceptance; issued separately |
| Converter and model integration | Numerical rules fixed, integration incomplete | block-FP8 encode entry, GLM converter/binding and the runtime path wait for implementation |

A quantization format exists does not mean a conversion method exists. The current block-FP8
converter has only `import_encoded`. The agreed `fp8_block_maxabs` must be implemented. The general
KDA entry serves algorithm and performance verification. It does not replace the call constraints of
the production ReplaySSM. Read the "to implement" scope of each design separately from the capability
of the existing general operators.

## 2. Current implementation scope and open design items

The existing base checklist covers only the pure projections, the stateless fusion, the dense/shared
MLP and the local Main/MTP MoE of TP2/EP2. The KDA state chain, MLA/Indexer and mHC are not appended
automatically. Implementation and tuning need an explicitly issued task. This document pass does not
change an entry ID, a status, a profile or an acceptance scope. It does not trigger the `.50` sync.

Work that still needs design or delivery:

- Cross-module collective, pack/unpack, compute-communication fusion, model Graph and cross-owner
  prefix restore.
- Local placement and interfaces for the MTP stem, the embedding and Vision; sharded logits,
  sampling, acceptance and scoring handoff.
- Full KDA input/output fusion and the execution choice for long prefill; the agreed recurrent
  constraints of decode/verify/fold stay.
- Converter, artifact binding, model load/Frontend/Engine integration and the complete state
  combination.

A design decision, a finished implementation and a verified performance result are three different
states. A finished operator kernel does not mean the model integration is finished. Local operator
performance does not mean a throughput gain for the two-host model.

## 3. Document responsibilities

| Content | Current sole owner |
|---|---|
| Model mathematics, logical roles and historical dependencies | Long-term model reference |
| Representation, codec, activation permission, observable rounding | Long-term numerical reference; an existing general format is referenced directly |
| Sharding, owners, communication handoff and deployment budget | parallel.md |
| Concrete Op/views, physical layout, write domain, alias, fusion and acceptance | The matching family design |
| Execution entries and completion evidence | op-checklist.md |
| Cross-topic progress and open scope | This index |

An interface table lists the dtype/shape it is consistent with. That table must reference and satisfy
the long-term semantics. Do not maintain two copies of an encoder, of the complete model mathematics
or of a different precision policy across documents. The mathematical combination of an
implementation design can reference the long-term formula, then state its own representation
boundary and state effect. Keep a necessary note where the sources differ. Git keeps the old
discussion history.

## 4. Takeover and retirement

| Design content | Authoritative location after it lands |
|---|---|
| Public input/output, alias, mathematical combination and state contract | The actual `include/ninfer/ops/` declaration |
| Encoding and physical layout | The general format/layout document and the matching implementation |
| Mathematics, exact commit and boundary acceptance | The matching qualification test |
| Current layer order, resources and Graph handoff | The model execution code and the `glm5next-execution.md` established at that time |
| Performance results | The matching benchmark/performance evidence |

After a part lands and passes acceptance, update the formal owner and the link, then remove the design
section that the implementation took over. While a cross-Op combination is not integrated, keep the
remaining design. Delete the file after every necessary part is taken over. Delete the checklist
after every item is complete and the evidence is in place. The long-term model and numerical choices
stay maintained. They do not disappear when a design retires.

Do not create an execution document here in advance that claims support. Do not keep an old path
alias or a coexisting old specification. Git keeps the history. The
[general distributed design](../../distributed-execution-design.md), the Qwen documents and the
existing engine contract stay maintained in their original locations.
