# PR #36 fix verification - results (eb9e87fa vs base ed6525fa)

## A: the fix tree (verify/pr36fix-2026-09-30)

## A-TESTS - ctest on verify/pr36fix-2026-09-30 (eb9e87fa + driver)

check 1, test_flash_next_qsa (exit 0; must pass on the fix):

1/1 Test #132: ninfer_flash_next_qsa_test .......   Passed    3.00 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =   3.01 sec

Flash-Next real suite (exit 8; the real test is expected to fail on the
stale golden until Opus re-records it; the other three tests must pass):

    Start 139: ninfer_qwen3_8_flash_next_load_plan_test
3/4 Test #139: ninfer_qwen3_8_flash_next_load_plan_test ...   Passed    0.08 sec
    Start 140: ninfer_qwen3_8_flash_next_frontend_test
4/4 Test #140: ninfer_qwen3_8_flash_next_frontend_test ....   Passed    1.28 sec

75% tests passed, 1 tests failed out of 4

Total Test time (real) = 124.78 sec

The following tests FAILED:
	134 - ninfer_qwen3_8_flash_next_real_test (Failed)
Errors while running CTest

mtp line from the real test:
mtp: 29108 4009 27891 8964 579 16078 321 1100 9872 303 660 17425
expected: 29108 4009 5435 660 7736 314 279 9155 19142 11 864 43000

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

## A-PPL - 4K PPL gate on verify/pr36fix-2026-09-30 (target 3.9982, run-to-run floor 1.3e-4)

exit 0

overall PPL 3.998162 (mean NLL 1.385835 over 1,044,557 tokens); delta vs 3.9982 = -3.75e-05 (within the 1.3e-4 floor)

log tail:

english_reference                 261223        1.405534        4.077704
ninfer_code                       259904        0.516196        1.675641
overall                          1044557        1.385835        3.998162

score rate: 772.7 tok/s
report: "/home/apollo11/ninfer-gb10/profiles/bench/gb10/pr36fix-verify/A-PPL/perplexity/report.json"

## B: the base tree (ed6525fa)

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

## B-PPL - 4K PPL gate on base ed6525fa

exit 0

overall PPL 3.998118 (mean NLL 1.385824 over 1,044,557 tokens); delta vs 3.9982 = -8.25e-05 (within the 1.3e-4 floor)

log tail:

english_reference                 261223        1.405417        4.077228
ninfer_code                       259904        0.516270        1.675765
overall                          1044557        1.385824        3.998118

score rate: 765.6 tok/s
report: "/home/apollo11/ninfer-gb10/profiles/bench/gb10/pr36fix-verify/B-PPL/perplexity/report.json"
