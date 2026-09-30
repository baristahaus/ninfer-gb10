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
