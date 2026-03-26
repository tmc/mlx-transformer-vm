/* Fibonacci: compute fib(n) where n is parsed from input string. */

void compute(const char *input) {
    int n;
    sscanf(input, "%d", &n);

    if (n == 0) { printf("0\n"); return; }
    if (n == 1) { printf("1\n"); return; }

    int a = 0, b = 1;
    int i;
    for (i = 2; i <= n; i++) {
        int t = (a + b);
        a = b;
        b = t;
    }

    printf("%d\n", b);
}
