pr36fix checks on /home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer (recipe qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp-v3)
ninfer: persistent grids sized for 48 SMs
mtp: 29108 4009 27891 8964 579 16078 321 1100 9872 303 660 17425
mtp backend=yes rounds=3 matches_fp8_mtp_golden=no matches_fp8_ordinary_golden=yes
PASS prefix reuse (reused=38, tokens:reuse: 3000 369
PASS custom stop + partial MTP terminal (reused=28)
concurrent root (normal):root: 29108 4009 27891 8964 579 16078 321 1100 9872 303 660 17425
NOTE root differs from the recorded golden (expected pre re-record)
concurrent root (reversed):root: 29108 4009 27891 8964 579 16078 321 1100 9872 303 660 17425
NOTE root differs from the recorded golden (expected pre re-record) (reversed admission)
PASS concurrent lane consistency (mtp line reused as root: root differs from the MTP greedy line)
near-tie index 2: token 27891 logprob -0.901279 top: 27891=-0.901279 5435=-1.27628 23716=-1.90128 1100=-2.40128 80291=-4.27628 12611=-4.40128 369=-4.52628 15783=-4.65128 69377=-5.15128 16661=-5.40128 18176=-5.52628 15734=-6.15128 21671=-6.46378 11=-6.52628 33060=-6.58878 7701=-7.27628 8222=-7.40128 35742=-7.58878 38666=-8.08878 41139=-8.15128
top-1/top-2 gap: 0.375 nats
near-tie mtp line: 29108 4009 27891 8964 579 16078 321 1100 9872 303 660 17425
OK pr36fix checks (0 failures)
