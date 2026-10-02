## Chunk-boundary memcheck

- Host: Linux 6.17.0-1029-nvidia aarch64; Ubuntu 24.04.4 LTS
- Commit: b624cc64 (with local changes)
- nvcc: release 13.0, V13.0.88
- GPU/driver: NVIDIA GB10, 580.173.02
- Artifact: qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer, 119G (multi-volume total)

| Part | MTP | Case | Exit | Sanitizer |
|---|---|---|---|---|
| A | mtp0 | prompt 255, chunk 256 | exit 0 | ERROR SUMMARY: 0 error |
| A | mtp0 | prompt 256, chunk 256 | exit 0 | ERROR SUMMARY: 0 error |
| A | mtp0 | prompt 257, chunk 256 | exit 0 | ERROR SUMMARY: 0 error |
| A | mtp0 | prompt 513, chunk 256 | exit 0 | ERROR SUMMARY: 0 error |
| A | mtp2 | prompt 255, chunk 256 | exit 0 | ERROR SUMMARY: 0 error |
| A | mtp2 | prompt 256, chunk 256 | exit 0 | ERROR SUMMARY: 0 error |
| A | mtp2 | prompt 257, chunk 256 | exit 0 | ERROR SUMMARY: 0 error |
| A | mtp2 | prompt 513, chunk 256 | exit 0 | ERROR SUMMARY: 0 error |
| B | mtp2 | 24 system lengths, chunk 128 | exit 0 | ERROR SUMMARY: 0 error |

B prompt lengths (the system-turn capture frontier moves by about one token per row):
```
0 prompt_tokens 149
1 prompt_tokens 150
2 prompt_tokens 152
3 prompt_tokens 153
4 prompt_tokens 154
5 prompt_tokens 157
6 prompt_tokens 158
7 prompt_tokens 159
8 prompt_tokens 160
9 prompt_tokens 162
10 prompt_tokens 163
11 prompt_tokens 164
12 prompt_tokens 165
13 prompt_tokens 166
14 prompt_tokens 168
15 prompt_tokens 169
16 prompt_tokens 171
17 prompt_tokens 174
18 prompt_tokens 176
19 prompt_tokens 178
20 prompt_tokens 179
21 prompt_tokens 181
22 prompt_tokens 182
23 prompt_tokens 184
```
