/* Long addition of two variable-length decimal numbers.
 * Input format: AAAA+BBBB  (no spaces, no leading zeros except "0" itself)
 * Output: the decimal sum followed by newline. */

void compute(const char *input) {
    /* Find the '+' separator and the end of input. */
    int plus_pos = 0;
    while (input[plus_pos] != '+') plus_pos++;
    int len = 0;
    while (input[len]) len++;
    int len_a = plus_pos;                /* digits of A: input[0..len_a-1] */
    int len_b = len - plus_pos - 1;      /* digits of B: input[plus_pos+1..] */

    /* Use the memory just past the null terminator of input as a result buffer. */
    char *buf = (char *)(input + len + 1);
    int max_len = (len_a > len_b ? len_a : len_b) + 1; /* +1 for possible carry */
    int pos = max_len;  /* index into buf, we fill backwards */

    int ia = len_a - 1;           /* index into A (rightmost digit) */
    int ib = plus_pos + len_b;    /* index into B (rightmost digit) */
    int carry = 0;

    while (ia >= 0 || ib > plus_pos || carry) {
        int da = 0;
        if (ia >= 0) { da = input[ia] - '0'; ia = ia - 1; }
        int db = 0;
        if (ib > plus_pos) { db = input[ib] - '0'; ib = ib - 1; }
        int sum = da + db + carry;
        carry = 0;
        if (sum >= 10) { sum = sum - 10; carry = 1; }
        pos = pos - 1;
        buf[pos] = '0' + sum;
    }

    /* Print the result digits. */
    while (pos < max_len) {
        putchar(buf[pos]);
        pos = pos + 1;
    }
    putchar('\n');
}
