float somefunc(float x) {
    for (int i = 0; i < 8; ++i) {
        if (i < 4) {
            continue;
        }
        else if (i == 5) {
            return x;
        }
    }
    return x;
}

