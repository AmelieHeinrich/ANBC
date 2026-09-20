#!/usr/bin/env python3
"""Dev-only check that src/anbc_vulkan.h is ABI-identical to the SDK's
vulkan_core.h: generates a C file that includes the real header, re-declares
every struct / enum of ours under an `A` prefix, and _Static_asserts
sizeof / offsetof of every field and every enum value. Compiled with the
host C compiler; any mismatch is a compile error.

    check_vulkan_abi.py [--sdk-include /usr/local/include] [--cc cc]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-include", type=Path, default=Path("/usr/local/include"))
    parser.add_argument("--cc", default="cc")
    args = parser.parse_args()

    ours = (ROOT / "src/anbc_vulkan.h").read_text()
    # Everything up to the loader section, with all Vk / VK / PFN identifiers prefixed.
    decls = ours[: ours.index("/* ----")]
    decls = decls[decls.index("#define VK_MAKE_API_VERSION"):]
    prefixed = re.sub(r"\b(Vk|VK_|PFN_vk)", r"A\1", decls)
    prefixed = prefixed.replace("AVK_DEFINE_HANDLE", "VK_DEFINE_HANDLE").replace(
        "AVK_DEFINE_NON_DISPATCHABLE_HANDLE", "VK_DEFINE_NON_DISPATCHABLE_HANDLE")

    structs = re.findall(r"typedef struct (Vk\w+) \{\n((?:[^}]*\n)*?)\} \1;", decls)
    unions = re.findall(r"typedef union (Vk\w+) \{\n((?:[^}]*\n)*?)\} \1;", decls)
    enums = re.findall(r"typedef enum (Vk\w+) \{\n((?:[^}]*\n)*?)\} \1;", decls)

    checks = []
    for name, body in structs + unions:
        checks.append(f"_Static_assert(sizeof({name}) == sizeof(A{name}), \"{name} size\");")
        for field in re.findall(r"^\s+[\w\s\*]+?\b(\w+)(?:\[[^\]]*\])*;", body, re.M):
            checks.append(f"_Static_assert(offsetof({name}, {field}) == offsetof(A{name}, {field}), \"{name}.{field}\");")
    for name, body in enums:
        checks.append(f"_Static_assert(sizeof({name}) == sizeof(A{name}), \"{name} size\");")
        for member in re.findall(r"^\s+(VK_\w+) = ", body, re.M):
            if member.endswith("_MAX_ENUM"):
                continue
            checks.append(f"_Static_assert((int){member} == (int)A{member}, \"{member}\");")

    src = ("#include <stddef.h>\n#include <vulkan/vulkan_core.h>\n#define ANBC_VULKAN_H\n"
           + prefixed + "\n" + "\n".join(checks) + "\nint main(void) { return 0; }\n")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "check_vulkan_abi.c"
        path.write_text(src)
        r = subprocess.run([args.cc, "-std=c11", "-fsyntax-only", f"-I{args.sdk_include}", str(path)])
        if r.returncode != 0:
            sys.exit("ABI mismatch between src/anbc_vulkan.h and the SDK header")
    print(f"anbc_vulkan.h ABI OK: {len(structs) + len(unions)} structs, {len(enums)} enums, {len(checks)} assertions")


if __name__ == "__main__":
    main()
