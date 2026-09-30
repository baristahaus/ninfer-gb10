## B-TESTS - base tree ed6525fa (the old golden must pass fully)

real suite (exit 0):

    Start 139: ninfer_qwen3_8_flash_next_load_plan_test
4/5 Test #139: ninfer_qwen3_8_flash_next_load_plan_test ...   Passed    0.04 sec
    Start 140: ninfer_qwen3_8_flash_next_frontend_test
5/5 Test #140: ninfer_qwen3_8_flash_next_frontend_test ....   Passed    0.62 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) = 153.87 sec

driver (exit 0):

pr36fix checks on /home/apollo11/ninfer-gb10/out/fp8mtp/qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp.ninfer (recipe qwen3_8_flash_next_125b_a6b_nvfp4_fp8_mtp-v3)
ninfer: persistent grids sized for 48 SMs
mtp: 29108 4009 5435 660 7736 314 279 9155 19142 11 864 43000
mtp backend=yes rounds=3 matches_fp8_mtp_golden=yes matches_fp8_ordinary_golden=no
PASS prefix reuse (reused=38, tokens:reuse: 3000 777
PASS custom stop + partial MTP terminal (reused=28)
concurrent root (normal):root: 29108 4009 5435 660 7736 314 279 9155 19142 11 864 43000
concurrent root (reversed):root: 29108 4009 5435 660 7736 314 279 9155 19142 11 864 43000
PASS concurrent lane consistency (mtp line reused as root: root equals the MTP greedy line)
near-tie index 2: token 5435 logprob -1.06157 top: 5435=-1.06157 27891=-1.06157 23716=-1.81157 1100=-2.68657 80291=-3.93657 12611=-4.56157 15783=-4.56157 18176=-5.06157 69377=-5.43657 11=-5.56157 369=-5.68657 16661=-5.68657 33060=-5.81157 15734=-6.18657 21671=-6.18657 8222=-6.62407 7701=-7.81157 35742=-7.87407 32785=-8.12407 18961=-8.24907
top-1/top-2 gap: 0 nats
near-tie mtp line: 29108 4009 5435 660 7736 314 279 9155 19142 11 864 43000
OK pr36fix checks (0 failures)
