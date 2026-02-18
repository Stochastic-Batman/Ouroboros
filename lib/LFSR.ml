(* LFSR.ml
   A Linear Feedback Shift Register (LFSR) is a shift register whose input bit
   is a linear function (XOR) of some of its previous state bits, called "taps".
   Despite producing sequences that look random, an LFSR is entirely deterministic
   and periodic — making it a great target for neural cryptanalysis. *)

(* A 32-bit maximal-length LFSR using the polynomial x**32 + x**22 + x**2 + x + 1.
   "Maximal length" means the sequence cycles through all 2**32 - 1 non-zero states
   before repeating. The taps are the exponents in the polynomial: 32, 22, 2, 1. *)
let taps = [32; 22; 2; 1]


(* [feedback state] computes the next input bit by XOR-ing together the bits
   at each tap position in [state]. This is the "linear" part of the LFSR. *)
let feedback (state : int64) : int64 =
  (* Fold over each tap, accumulating the XOR of those bit positions. *)
  List.fold_left
    (fun acc tap ->
      (* Shift state right so the tap bit lands at position 0, then mask it. *)
      let bit = Int64.logand (Int64.shift_right_logical state (tap - 1)) 1L in
      Int64.logxor acc bit)
    0L
    taps


(* [next_state state] advances the LFSR by one step.
   The register shifts right by one, and the new feedback bit is placed
   into the most significant bit (bit 31 for a 32-bit register). *)
let next_state (state : int64) : int64 =
  let fb  = feedback state in
  (* Shift existing bits right, drop the outgoing LSB. *)
  let shifted = Int64.shift_right_logical state 1 in
  (* Place the feedback bit at position 31 (the MSB of a 32-bit window). *)
  Int64.logor shifted (Int64.shift_left fb 31)


(* [output_bit state] returns the current output bit of the LFSR,
   which is simply the least significant bit of the state. *)
let output_bit (state : int64) : int =
  Int64.to_int (Int64.logand state 1L)


(* [generate_sequence seed n] produces a list of [n] output bits starting from
   [seed], advancing the LFSR state for each bit. This is the raw training data
   that the RNN will try to predict. *)
let generate_sequence (seed : int64) (n : int) : int list =
  (* Use a recursive helper that threads the current state through each step. *)
  let rec loop state count acc =
    if count = 0 then List.rev acc
    else
      let bit   = output_bit state in
      let state' = next_state state in
      loop state' (count - 1) (bit :: acc)
  in
  loop seed n []
