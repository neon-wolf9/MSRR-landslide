# Optional third-party source checkout

No third-party comparator source is distributed in this repository.

To execute the original PU-pullbaggingDT provenance guard, place an independently obtained checkout at:

`third_party/PU-pullbaggingDT`

Required upstream repository: `https://github.com/ShubingOuyangcug/PU-pullbaggingDT`

Required commit: `a8857dad30a454e3155644b18ddd2600df860e70`

An independent user may clone the upstream repository, check out the stated commit, and verify `git rev-parse HEAD` and a clean `git status --porcelain` before placing it at that path. The inspected upstream copy contained no explicit license file. Public access does not grant redistribution rights; do not include its source in this MIT-licensed package unless the rights holder supplies permission.

The `128B1` script uses this checkout for a commit/clean-tree provenance guard. It does **not** import or execute any of the upstream Python files; the paper comparator is implemented by the separately disclosed author adapter. Reviewers should not interpret the checkout as proof that the unmodified upstream algorithm was run. Frozen published-comparator summaries remain inspectable without the checkout, but formal comparison still requires it to satisfy the guard.
