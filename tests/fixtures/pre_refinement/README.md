# Frozen pre-refinement simulator reference

This is a test-only copy of the first-party synthetic simulator and its two
configuration inputs from commit
`881f6aa9ffcd7ce3928a5c68ad5b71e9129ec231`, the parent of the refinement change
`ffb314dd736a750e026861284e437aafa1e935ee`. The 15 source/configuration files are
individually bound by `provenance.json` and remain covered
by the repository's MIT license. Only the package docstring was updated during
submission cleanup to match the title and remove an obsolete directory reference;
its original hash and the change are recorded in the manifest. All executable
statements and the other 14 files remain verbatim Git blobs. This fixture contains no market data, training
code, model weights or paper results. Historical-input loader functions are
retained verbatim as part of the source modules but are never called by this test.

The regression runs a single synthetic no-op episode (seed 4201) in an isolated
subprocess using this source and the same interpreter/dependencies as the current
simulator. It compares all 71 observations, rewards, decision times, H_used/H_next
values and the final RNG state exactly. It does not round outputs or allow a
numerical tolerance. The test works in a source archive without Git or network
access; it checks the frozen source hashes before executing it.

The former assertion hashed JSON floats against one captured runtime. Replaying
the original source reproduced that original hash on Python 3.14.4 / NumPy 2.4.6
on macOS arm64. Both the original and current simulators instead produced the
same different hash on Python 3.10.6 / NumPy 1.26.4 on that machine; the latter is
an additional diagnostic environment outside the declared NumPy >=2 dependency
range. These checks establish exact preservation across the refinement change
within each tested runtime, without claiming cross-runtime bitwise identity.
Python 3.12 also changed floating-point `sum`, which is used by Algorithm 1;
see the [Python documentation](https://docs.python.org/3/library/functions.html#sum).
The complete cause of the separately reported Linux hash has not been isolated.

This reference deliberately retains the original implementation, including its
limitations. It checks preservation, not scientific correctness or independent
replication of the paper. Do not refresh it from the current simulator to silence
a failure. Any intentional change to the compatibility contract needs a separately
reviewed explanation and reference revision.
