// Copyright 2026 Percepta
// Licensed under the Apache License, Version 2.0.
// Obtained from https://github.com/Percepta-Core/transformer-vm
// SPDX-License-Identifier: Apache-2.0

/* Hello: greet the user by name. */

void compute(const char *input) {
    print_str("Hello ");
    print_str(input);
    print_str("!\n");
}
