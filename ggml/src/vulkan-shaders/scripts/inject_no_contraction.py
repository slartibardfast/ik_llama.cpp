#!/usr/bin/env python3
"""Post-process a SPIR-V assembly file to add NoContraction decoration to
every OpFMul and OpFAdd result id. This prevents the driver-level compiler
(ACO for RADV) from fusing mul+add into FMA with variable precision across
pipeline specializations.

Usage: inject_no_contraction.py input.spvasm output.spvasm
"""
import re
import sys

def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path) as f:
        lines = f.readlines()

    # Find every "%id = OpFMul" or "%id = OpFAdd" — collect result ids.
    pat = re.compile(r'^\s*(%[A-Za-z0-9_]+)\s*=\s*OpF(Mul|Add|Sub|Negate)\b')
    result_ids = []
    for ln in lines:
        m = pat.match(ln)
        if m:
            result_ids.append(m.group(1))

    print(f"[inject] found {len(result_ids)} OpFMul/OpFAdd/OpFSub/OpFNegate results", file=sys.stderr)

    # Build decoration lines to inject.
    decs = [f"               OpDecorate {rid} NoContraction\n" for rid in result_ids]

    # Insert decorations after the last OpDecorate/OpMemberDecorate line in the
    # Annotations section, before non-decoration code starts.
    # Find the last decoration line.
    last_dec_idx = -1
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith('OpDecorate') or stripped.startswith('OpMemberDecorate'):
            last_dec_idx = i

    if last_dec_idx == -1:
        print("[inject] no OpDecorate found — inserting before %1 = OpExtInstImport", file=sys.stderr)
        for i, ln in enumerate(lines):
            if 'OpExtInstImport' in ln or 'OpTypeVoid' in ln or 'OpCapability' in ln:
                last_dec_idx = i - 1
                break

    new_lines = lines[:last_dec_idx+1] + decs + lines[last_dec_idx+1:]
    with open(out_path, 'w') as f:
        f.writelines(new_lines)
    print(f"[inject] wrote {out_path} with {len(decs)} NoContraction decorations added", file=sys.stderr)

if __name__ == '__main__':
    main()
