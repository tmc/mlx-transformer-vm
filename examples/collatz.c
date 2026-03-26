/* Collatz conjecture: given n, output the sequence until it reaches 1.
 */

void compute(const char *input) {
    int n;
    sscanf(input, "%d", &n);
    if (n <= 0) { printf("need n>0\n"); return; }

    printf("%d", n);
    while (n != 1) {
        /* Check if n is even */
        if (n % 2 == 0) {
            /* n is even: n = n / 2 */
            n = n / 2;
        } else {
            /* n is odd: n = 3*n + 1 */
            n = 3 * n + 1;
        }
        printf(" %d", n);
    }
    printf("\n");
}
