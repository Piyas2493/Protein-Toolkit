Downloaded external tool binaries go here (gitignored — not source).

## AutoDock Vina

`vina.exe` (Windows) — real docking backend for `docking/engines/vina_adapter.py`.
Not pip-installable on native Windows (needs Boost + a C++ toolchain to build
from source). Get the precompiled binary instead:

    https://github.com/ccsb-scripps/AutoDock-Vina/releases

Download `vina_<version>_win.exe`, save it here as `bin/vina.exe`. The
adapter checks PATH first, then falls back to this folder automatically.
