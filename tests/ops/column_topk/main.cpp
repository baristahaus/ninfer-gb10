#include "ops/suite.h"

void add_column_topk_cases(ninfer::test::OpSuite&);

int main(int argc, char** argv) {
    ninfer::test::OpSuite suite;
    add_column_topk_cases(suite);
    return suite.main(argc, argv);
}
