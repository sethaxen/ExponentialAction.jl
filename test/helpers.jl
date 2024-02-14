expv_explicit(t, A, B) = exp(t * A) * B

expv_sequence_explicit(ts, A, B) = map(t -> expv_explicit(t, A, B), ts)
