#!/usr/bin/env python3
"""Drop-in `jac` CLI with small client-bundle fixes for current Jaclang output.

- Mirror exported client functions onto ``globalThis`` before registration (ES modules).
- Drop duplicate ``const _jac`` blocks from the merged bundle.
- Rewrite ``.append(`` to ``.push(`` so hooks/signals work in the browser.
- Unwrap ``__jacCallFunction`` results with ``hasOwnProperty("result")`` so JSON null / 0 / ``""`` are not dropped.
- If the server returns ``{ "error": ... }`` without ``result`` (Jac still uses HTTP 200), throw so the UI shows an error instead of silent ``null``.

Run from this directory: ``python jac.py start main.jac`` (same subcommands as ``jac``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

# Bundler inlines `@jac/runtime` and may inject the same `const _jac = { ... }` again
# when the app module sets `needs_jac_runtime` (e.g. broad `except Exception`), which
# breaks the browser with "Identifier '_jac' has already been declared".
_JAC_CONST_MARKER = "const _jac = {"


def _remove_second_jac_runtime_block(code: str) -> str:
    first = code.find(_JAC_CONST_MARKER)
    if first == -1:
        return code
    second = code.find(_JAC_CONST_MARKER, first + len(_JAC_CONST_MARKER))
    if second == -1:
        return code
    brace_open = second + len(_JAC_CONST_MARKER) - 1
    if brace_open >= len(code) or code[brace_open] != "{":
        return code
    depth = 0
    i = brace_open
    while i < len(code):
        c = code[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                i += 1
                if i < len(code) and code[i] == ";":
                    i += 1
                while i < len(code) and code[i] in "\r\n":
                    i += 1
                return code[:second] + code[i:]
        i += 1
    return code


def _dedupe_jac_runtime_blocks(code: str) -> str:
    while code.count(_JAC_CONST_MARKER) > 1:
        nxt = _remove_second_jac_runtime_block(code)
        if nxt == code:
            break
        code = nxt
    return code


def _fix_client_bundle_list_append(code: str) -> str:
    """Jac's client JS still emits ``.append(`` for lists; Arrays need ``.push(``."""
    return code.replace(".append(", ".push(")


def _fix_jac_call_function_result_guard(code: str) -> str:
    """``if (response_data[\"result\"])`` drops JSON null / 0 / \"\" and hides server errors."""
    old = """  try {
    if (response_data["result"]) {
      result = response_data["result"];
    }
  } catch (__jac_e) {"""
    new = """  try {
    if (Object.prototype.hasOwnProperty.call(response_data, "result")) {
      result = response_data["result"];
    }
  } catch (__jac_e) {"""
    return code.replace(old, new) if old in code else code


def _fix_jac_call_function_error_envelope(code: str) -> str:
    """Surface ``data.error`` when there is no ``result`` (e.g. ImportError in a @pub fn)."""
    old = """  let response_data = (payload["data"] ? payload["data"] : payload);
  let result = null;"""
    new = """  let response_data = (payload["data"] ? payload["data"] : payload);
  if (
    response_data &&
    Object.prototype.hasOwnProperty.call(response_data, "error") &&
    response_data["error"] != null &&
    !Object.prototype.hasOwnProperty.call(response_data, "result")
  ) {
    const err = response_data["error"];
    throw new _jac.exc.Exception(
      typeof err === "string" ? err : JSON.stringify(err)
    );
  }
  let result = null;"""
    return code.replace(old, new) if old in code else code


def _apply_client_bundle_global_export_fix() -> None:
    @staticmethod
    def _generate_registration_js(
        module_name: str,
        client_functions: Sequence[str],
        client_globals: dict[str, Any],
    ) -> str:
        globals_entries: list[str] = []
        for name, value in client_globals.items():
            identifier = json.dumps(name)
            try:
                value_literal = json.dumps(value)
            except TypeError:
                value_literal = "null"
            globals_entries.append(f"{identifier}: {value_literal}")
        globals_literal = (
            "{ " + ", ".join(globals_entries) + " }" if globals_entries else "{}"
        )
        functions_literal = json.dumps(list(client_functions))
        module_literal = json.dumps(module_name)
        assign_lines = [
            f"globalThis[{json.dumps(name)}] = {name};" for name in client_functions
        ]
        assigns_prefix = ("\n".join(assign_lines) + "\n") if assign_lines else ""
        return (
            f"{assigns_prefix}__jacRegisterClientModule({module_literal}, {functions_literal}, {globals_literal});"
        )

    import jaclang.runtimelib.client_bundle as client_bundle

    client_bundle.ClientBundleBuilder._generate_registration_js = (  # type: ignore[method-assign]
        _generate_registration_js  # noqa: SLF001 — replaces Jac-generated static
    )


def _apply_client_bundle_jac_dedupe_fix() -> None:
    import jaclang.runtimelib.client_bundle as client_bundle

    Builder = client_bundle.ClientBundleBuilder
    _orig_build = Builder.build

    def build(self, module, force: bool = False):  # type: ignore[no-untyped-def]
        bundle = _orig_build(self, module, force)
        new_code = _fix_jac_call_function_error_envelope(
            _fix_jac_call_function_result_guard(
                _fix_client_bundle_list_append(_dedupe_jac_runtime_blocks(bundle.code))
            )
        )
        if new_code != bundle.code:
            bundle.code = new_code
            bundle.hash = hashlib.sha256(new_code.encode("utf-8")).hexdigest()
        return bundle

    Builder.build = build  # type: ignore[method-assign]


def main() -> None:
    _apply_client_bundle_global_export_fix()
    _apply_client_bundle_jac_dedupe_fix()
    from jaclang.jac0core.cli_boot import start_cli

    start_cli()


if __name__ == "__main__":
    main()
