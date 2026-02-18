(* test_ouroboros.ml
   Unit tests for the core math in both LFSR generation and the RNN. *)

open Owl
module M = Dense.Matrix.S


(* -------------------------------------------------------------------------- *)
(* Tiny test framework                                                          *)
(* -------------------------------------------------------------------------- *)

let pass_count = ref 0
let fail_count = ref 0


(* [check name condition] prints PASS/FAIL and updates counters. *)
let check (name : string) (condition : bool) =
  if condition then begin
    Printf.printf "[PASS] %s\n" name;
    incr pass_count
  end else begin
    Printf.printf "[FAIL] %s\n" name;
    incr fail_count
  end


(* -------------------------------------------------------------------------- *)
(* LFSR tests                                                                  *)
(* -------------------------------------------------------------------------- *)

(* The sequence must be deterministic: same seed → same bits. *)
let test_lfsr_deterministic () =
  let s1 = LFSR.generate_sequence 1L 100 in
  let s2 = LFSR.generate_sequence 1L 100 in
  check "LFSR is deterministic" (s1 = s2)


(* All output bits must be 0 or 1. *)
let test_lfsr_bits_are_binary () =
  let bits = LFSR.generate_sequence 42L 200 in
  let all_binary = List.for_all (fun b -> b = 0 || b = 1) bits in
  check "LFSR outputs only 0/1" all_binary


(* A 32-bit maximal-length LFSR must eventually change state. *)
let test_lfsr_not_constant () =
  let bits = LFSR.generate_sequence 1L 64 in
  let has_zero = List.exists (fun b -> b = 0) bits in
  let has_one  = List.exists (fun b -> b = 1) bits in
  check "LFSR produces both 0 and 1" (has_zero && has_one)


(* Different seeds must produce different sequences. *)
let test_lfsr_different_seeds () =
  let s1 = LFSR.generate_sequence 1L 50 in
  let s2 = LFSR.generate_sequence 2L 50 in
  check "Different seeds → different sequences" (s1 <> s2)


(* The all-zeros state is a fixed point and should stay constant.
   (The zero state is the degenerate case for XOR-based LFSRs.) *)
let test_lfsr_zero_seed_fixed () =
  let bits = LFSR.generate_sequence 0L 10 in
  check "Zero seed is a fixed point (all zeros)" (List.for_all (fun b -> b = 0) bits)


(* -------------------------------------------------------------------------- *)
(* Model tests                                                                  *)
(* -------------------------------------------------------------------------- *)

(* A freshly created model should have the right parameter shapes. *)
let test_model_shapes () =
  let m = Model.create 16 in
  check "wxh shape [16×1]"   (M.row_num m.wxh = 16 && M.col_num m.wxh = 1);
  check "whh shape [16×16]"  (M.row_num m.whh = 16 && M.col_num m.whh = 16);
  check "bh  shape [16×1]"   (M.row_num m.bh  = 16 && M.col_num m.bh  = 1);
  check "why shape [1×16]"   (M.row_num m.why = 1  && M.col_num m.why = 16);
  check "by  shape [1×1]"    (M.row_num m.by  = 1  && M.col_num m.by  = 1)


(* A single forward step must return a probability in (0, 1). *)
let test_forward_output_range () =
  let m  = Model.create 8 in
  let h0 = Model.zero_hidden m in
  let (_h, y, _cache) = Model.forward m h0 1.0 in
  check "Forward output in (0,1)" (y > 0.0 && y < 1.0)


(* BCE loss should be positive for any prediction ≠ target. *)
let test_bce_loss_positive () =
  let loss = Model.bce_loss 0.7 0.0 in
  check "BCE loss > 0 for wrong prediction" (loss > 0.0)


(* BCE loss should be near zero for a nearly perfect prediction. *)
let test_bce_loss_near_zero () =
  let loss = Model.bce_loss 0.9999 1.0 in
  check "BCE loss ≈ 0 for correct prediction" (loss < 0.01)


(* Gradient clipping: no element should exceed the clip value. *)
let test_gradient_clipping () =
  let m = Model.create 4 in
  (* Build a grads record with large values. *)
  let big = M.create 4 1 999.0 in
  let g : Model.grads = {
    d_wxh = big; d_whh = M.create 4 4 999.0;
    d_bh  = big; d_why = M.create 1 4 999.0;
    d_by  = M.create 1 1 999.0;
  } in
  let clipped = Model.clip_grads g 5.0 in
  let max_val = M.max' clipped.d_wxh in
  check "Gradient clipping caps values at clip_val" (max_val <= 5.0)


(* -------------------------------------------------------------------------- *)
(* Run all tests                                                                *)
(* -------------------------------------------------------------------------- *)

let () =
  Printf.printf "\n=== LFSR Tests ===\n";
  test_lfsr_deterministic ();
  test_lfsr_bits_are_binary ();
  test_lfsr_not_constant ();
  test_lfsr_different_seeds ();
  test_lfsr_zero_seed_fixed ();

  Printf.printf "\n=== Model Tests ===\n";
  test_model_shapes ();
  test_forward_output_range ();
  test_bce_loss_positive ();
  test_bce_loss_near_zero ();
  test_gradient_clipping ();

  Printf.printf "\n=== Results: %d passed, %d failed ===\n"
    !pass_count !fail_count;

  (* Exit with a non-zero code if any test failed so CI can catch it. *)
  if !fail_count > 0 then exit 1
