(* train.ml
   Orchestrates training: slices the LFSR bitstream into fixed-length chunks,
   runs a forward pass over each chunk (collecting caches), then unrolls the
   backward pass through time (BPTT) to accumulate gradients, and finally
   updates the model weights. *)

open Owl
module M = Dense.Matrix.S


(* Sequence-level forward pass                                                  *)

(* [forward_sequence model h0 inputs targets] runs the RNN over a list of
   input bits, predicting the next bit at each step.
   Returns:
     total_loss - summed BCE over the sequence
     caches     - list of per-step caches (in forward order) for BPTT
     preds      - list of per-step predictions (for accuracy reporting)
     h_last     - hidden state at the end, so state can carry across chunks *)
let forward_sequence (model : Model.t) (h0 : M.mat) (inputs  : float list) (targets : float list) : float * (M.mat * M.mat * float) list * float list * M.mat =
  (* Zip inputs and targets so we can iterate step by step. *)
  let pairs = List.combine inputs targets in

  let (loss, caches, preds, h_last) =
    List.fold_left
      (fun (acc_loss, acc_caches, acc_preds, h_prev) (x, y_true) ->
        let (h_next, y_pred, cache) = Model.forward model h_prev x in
        let step_loss = Model.bce_loss y_pred y_true in
        ( acc_loss +. step_loss,
          cache :: acc_caches,     (* prepend; reverse later for BPTT *)
          y_pred :: acc_preds,
          h_next ))
      (0.0, [], [], h0)
      pairs
  in
  (* Caches were prepended so they're currently in reverse (newest first),
     which is exactly what we want for BPTT (backward from last step). *)
  (loss /. float_of_int (List.length inputs), caches, List.rev preds, h_last)


(* Sequence-level backward pass (BPTT)                                          *)

(* [add_grads a b] element-wise sums two gradient records — used to accumulate
   gradients across all time steps before applying a single update. *)
let add_grads (a : Model.grads) (b : Model.grads) : Model.grads =
  { Model.d_wxh = M.add a.d_wxh b.d_wxh;
    d_whh       = M.add a.d_whh b.d_whh;
    d_bh        = M.add a.d_bh  b.d_bh;
    d_why       = M.add a.d_why b.d_why;
    d_by        = M.add a.d_by  b.d_by; }


(* [zero_grads model] creates a gradient record filled with zeros — the
   accumulator that we sum into during BPTT. *)
let zero_grads (model : Model.t) : Model.grads =
  { Model.d_wxh = M.zeros (M.row_num model.wxh) (M.col_num model.wxh);
    d_whh       = M.zeros (M.row_num model.whh) (M.col_num model.whh);
    d_bh        = M.zeros (M.row_num model.bh)  (M.col_num model.bh);
    d_why       = M.zeros (M.row_num model.why) (M.col_num model.why);
    d_by        = M.zeros (M.row_num model.by)  (M.col_num model.by); }


(* [backward_sequence model caches preds targets clip_val] runs BPTT over
   all steps.  Caches are already in reverse order (newest first), so we walk
   backwards naturally.  Returns the accumulated, clipped gradient record. *)
let backward_sequence (model    : Model.t) (caches   : (M.mat * M.mat * float) list) (preds    : float list) (targets  : float list) (clip_val : float) : Model.grads =

  (* Reverse preds/targets to match the newest-first order of caches. *)
  let preds_rev   = List.rev preds   in
  let targets_rev = List.rev targets in
  let triples     = List.combine caches (List.combine preds_rev targets_rev) in

  let (acc_grads, _dh_next) =
    List.fold_left
      (fun (acc_g, dh_next) (cache, (y_pred, y_true)) ->
        (* d_loss/d_y_raw for BCE+sigmoid fused: y_pred - y_true *)
        let dy = y_pred -. y_true in
        let (step_grads, dh_prev) = Model.backward model cache dh_next dy in
        ( add_grads acc_g step_grads, dh_prev ))
      (zero_grads model, Model.zero_hidden model)
      triples
  in
  Model.clip_grads acc_grads clip_val


(* Accuracy helper                                                              *)

(* [accuracy preds targets] returns the fraction of predictions that round to
   the correct bit.  A prediction >= 0.5 is interpreted as "1". *)
let accuracy (preds : float list) (targets : float list) : float =
  let n = List.length preds in
  let correct =
    List.fold_left2
      (fun acc p t ->
        let predicted_bit = if p >= 0.5 then 1.0 else 0.0 in
        if Float.equal predicted_bit t then acc + 1 else acc)
      0 preds targets
  in
  float_of_int correct /. float_of_int n


(* Main training loop                                                           *)

(* [train ~seed ~hidden_size ~seq_len ~n_steps ~lr ~momentum ~clip_val] trains
   an RNN to predict LFSR output bits.

   Parameters:
     seed        - the initial LFSR state (determines the bitstream)
     hidden_size - number of hidden units in the RNN
     seq_len     - how many bits to unroll per gradient step (BPTT window)
     n_steps     - total number of gradient steps to take
     lr          - SGD learning rate
     momentum    - SGD momentum coefficient (0 = vanilla SGD, 0.9 = typical)
     clip_val    - gradient clipping threshold                               *)
let train ?(seed = 0xDEADBEEFL) ?(hidden_size = 32) ?(seq_len = 64) ?(n_steps = 2000) ?(lr = 0.01) ?(momentum = 0.9) ?(clip_val = 5.0) () =

  Printf.printf "Ouroboros — Neural Cryptanalysis of a 32-bit LFSR\n";
  Printf.printf "hidden=%d  seq_len=%d  lr=%.4f  momentum=%.2f  steps=%d\n\n%!"
    hidden_size seq_len lr momentum n_steps;

  (* Generate a long bitstream upfront — we'll slice windows from it. *)
  let total_bits = n_steps * seq_len + seq_len in
  let bits = Array.of_list (LFSR.generate_sequence seed total_bits) in

  let model = Model.create hidden_size in

  (* We carry the hidden state across consecutive chunks so the RNN can
     learn long-range structure beyond a single BPTT window. *)
  let h0  = Model.zero_hidden model in

  (* Velocity starts at zero — it will warm up over the first few steps as
     the momentum term accumulates gradient history. *)
  let vel = Model.zero_velocity model in

  (* Fold over each gradient step, threading (model, hidden_state, velocity)
     through so all three evolve together without mutable state. *)
  let _final =
    List.fold_left
      (fun (mdl, h, vel) step ->
        (* Slice [step*seq_len .. step*seq_len + seq_len] from the stream. *)
        let offset      = step * seq_len in
        let window      = Array.sub bits offset seq_len |> Array.to_list in
        (* Targets are the bits one position ahead of the inputs. *)
        let next_window = Array.sub bits (offset + 1) seq_len |> Array.to_list in

        (* Convert int bits to floats for the network. *)
        let inputs  = List.map float_of_int window in
        let targets = List.map float_of_int next_window in

        if List.length inputs < seq_len || List.length targets < seq_len then
          (mdl, h, vel)  (* ran off the end of the pre-generated stream *)
        else begin
          let (loss, caches, preds, h_last) =
            forward_sequence mdl h inputs targets in

          let grads        = backward_sequence mdl caches preds targets clip_val in
          let (mdl', vel') = Model.update mdl grads vel lr momentum in

          (* Print progress every 100 steps. *)
          if (step + 1) mod 100 = 0 then begin
            let acc = accuracy preds targets in
            Printf.printf "step %4d/%d  loss=%.4f  acc=%.2f%%\n%!"
              (step + 1) n_steps loss (acc *. 100.0)
          end;

          (* Detach hidden state - clamp it so it can't drift outside tanh range *)
          let h_last_detached = M.map (fun v -> Float.max (-1.0) (Float.min 1.0 v)) h_last in
          (mdl', h_last_detached, vel')
        end)
      (model, h0, vel)
      (List.init n_steps (fun i -> i))
  in
  Printf.printf "\nDone.\n"
