# Source modules

`src/benchmark/` contains copies of the underlying matching, common-support, and spatial-split authority modules. The formal experiment drivers remain under `scripts/` with their original experiment identifiers because several scripts import one another by those filenames.

The other `src/` category directories are reserved navigation points. They are not populated with rewritten abstractions: the package deliberately avoids replacing the executed monolithic authority scripts with cleaner but scientifically different code.
