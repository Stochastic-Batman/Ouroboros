(* model.ml
   A minimal Elman RNN (simple recurrent network). At each time step t the
   network reads one input bit x_t, combines it with the previous hidden state
   h_{t-1}, and emits a probability that the *next* bit will be 1.

   Equations:
     h_t = tanh( Wxh * x_t + Whh * h_{t-1} + bh )
     y_t = sigmoid( Why * h_t + by )

   All weight matrices are plain Owl dense float arrays so we can manage
   gradients by hand — this keeps the code transparent and dependency-light. *)

open Owl

(* Convenience alias for the Owl dense float matrix module. *)
module M = Dense.Matrix.S


(* Model record                                                                 *)

(* [t] bundles every learnable parameter together so we can pass the whole
   model around as a single value. *)
type t = {
  wxh : M.mat;   (* input  -> hidden weights,  shape [hidden_size x 1]          *)
  whh : M.mat;   (* hidden -> hidden weights,  shape [hidden_size x hidden_size] *)
  bh  : M.mat;   (* hidden bias,              shape [hidden_size x 1]           *)
  why : M.mat;   (* hidden -> output weights,  shape [1 x hidden_size]           *)
  by  : M.mat;   (* output bias,              shape [1 x 1]                     *)
  hidden_size : int;
}


(* Initialisation                                                               *)

(* [create hidden_size] allocates a fresh model with Xavier-style random weights.
   Xavier init scales weights by 1/sqrt(fan_in) to keep activations from
   exploding or vanishing at the start of training. *)
let create (hidden_size : int) : t =
  let fan_wxh = float_of_int 1 in          (* input size is 1 (one bit)        *)
  let fan_whh = float_of_int hidden_size in

  (* Helper: random matrix with uniform values scaled by 1/sqrt(fan_in). *)
  let rand_mat rows cols fan =
    let scale = 1.0 /. sqrt fan in
    let m = M.uniform rows cols in
    (* Centre around zero: uniform[0,1] → uniform[-scale, scale]. *)
    M.map (fun v -> (v -. 0.5) *. 2.0 *. scale) m
  in

  { wxh         = rand_mat hidden_size 1 fan_wxh;
    whh         = rand_mat hidden_size hidden_size fan_whh;
    bh          = M.zeros hidden_size 1;
    why         = rand_mat 1 hidden_size fan_whh;
    by          = M.zeros 1 1;
    hidden_size; }


(* Activation functions                                                         *)

(* [sigmoid x] squashes any real value into (0, 1) — used for the output
   because we want a probability. *)
let sigmoid x = 1.0 /. (1.0 +. exp (-. x))


(* Element-wise sigmoid over a matrix. *)
let sigmoid_mat m = M.map sigmoid m


(* Element-wise tanh — Owl wraps this for us. *)
let tanh_mat m = M.map tanh m


(* Forward pass                                                                 *)

(* [forward model h_prev x_bit] runs a single time-step of the RNN.
   Returns:
     h_next  - the new hidden state (to be fed into the next step)
     y       - the predicted probability that the next output bit is 1
     cache   - intermediate values saved for the backward pass           *)
let forward (model : t) (h_prev : M.mat) (x_bit : float) : M.mat * float * (M.mat * M.mat * float) =
  (* Wrap the scalar input as a 1×1 matrix so Owl can multiply it. *)
  let x = M.of_array [| x_bit |] 1 1 in

  (* h_t = tanh( Wxh·x + Whh·h_{t-1} + bh ) *)
  let h_raw = M.add (M.add (M.dot model.wxh x) (M.dot model.whh h_prev)) model.bh in
  let h_next = tanh_mat h_raw in

  (* y_t = sigmoid( Why·h_t + by ) - scalar probability *)
  let y_raw = M.add (M.dot model.why h_next) model.by in
  let y     = sigmoid (M.get y_raw 0 0) in

  (h_next, y, (h_prev, h_next, x_bit))


(* Loss                                                                         *)

(* [bce_loss y_pred y_true] computes binary cross-entropy for a single
   prediction.  BCE = -[ y·log(p) + (1-y)·log(1-p) ]
   A small epsilon is added inside the log to avoid log(0). *)
let bce_loss (y_pred : float) (y_true : float) : float =
  let eps = 1e-7 in
  -. (y_true *. log (y_pred +. eps) +. (1.0 -. y_true) *. log (1.0 -. y_pred +. eps))


(* Backward pass (BPTT for a single step)                                       *)

(* Gradients with respect to every parameter, mirroring the [t] record. *)
type grads = {
  d_wxh : M.mat;
  d_whh : M.mat;
  d_bh  : M.mat;
  d_why : M.mat;
  d_by  : M.mat;
}


(* [backward model cache dh_next dy] computes gradients for one time-step.
   [dh_next] is the gradient flowing back from the *next* time-step (zero on
   the last step); [dy] is d_loss/d_y for this step. *)
let backward (model : t) (cache  : M.mat * M.mat * float) (dh_next : M.mat) (dy : float) : grads * M.mat =
  let (h_prev, h_next, x_bit) = cache in
  let x = M.of_array [| x_bit |] 1 1 in

  (* Output layer gradients *)
  (* d_loss/d_y_raw = dy * sigmoid'(y_raw).  Since sigmoid'(z) = y*(1-y) and
     we already have the sigmoid output in y, this simplifies to just dy when
     we absorb the sigmoid derivative into the BCE gradient (the two cancel). *)
  let dy_raw = M.of_array [| dy |] 1 1 in

  (* d_why:  outer product  dy_raw · h_next^T, shape [1 x hidden_size] *)
  let d_why = M.dot dy_raw (M.transpose h_next) in
  let d_by  = dy_raw in

  (* Gradient w.r.t. h_next from the output layer. *)
  let dh_from_y = M.dot (M.transpose model.why) dy_raw in

  (* Total gradient into h_next (from output + next time-step). *)
  let dh = M.add dh_from_y dh_next in

  (* Hidden layer gradients *)
  (* tanh'(z) = 1 - tanh(z)^2.  h_next = tanh(h_raw), so: *)
  let dtanh = M.map (fun h -> 1.0 -. h *. h) h_next in
  let dh_raw = M.mul dh dtanh in

  let d_wxh = M.dot dh_raw (M.transpose x) in
  let d_whh = M.dot dh_raw (M.transpose h_prev) in
  let d_bh  = dh_raw in

  (* Propagate gradient back to the previous hidden state. *)
  let dh_prev = M.dot (M.transpose model.whh) dh_raw in

  ( { d_wxh; d_whh; d_bh; d_why; d_by }, dh_prev )


(* Parameter update (SGD with momentum and gradient clipping)                  *)

(* [clip_grads g clip_val] clips every gradient matrix so that no element
   exceeds [clip_val] in absolute value.  Clipping prevents exploding gradients
   which are common in RNNs on long sequences. *)
let clip_grads (g : grads) (clip_val : float) : grads =
  let clip m = M.map (fun v -> Float.max (-. clip_val) (Float.min clip_val v)) m in
  { d_wxh = clip g.d_wxh;
    d_whh = clip g.d_whh;
    d_bh  = clip g.d_bh;
    d_why = clip g.d_why;
    d_by  = clip g.d_by; }


(* Velocity record — one matrix per parameter, mirroring [t].
   Momentum accumulates a running average of past gradients so the optimiser
   can build up speed in consistent directions and dampen oscillations. *)
type velocity = {
  v_wxh : M.mat;
  v_whh : M.mat;
  v_bh  : M.mat;
  v_why : M.mat;
  v_by  : M.mat;
}


(* [zero_velocity model] initialises all velocity matrices to zero — called
   once before training begins. *)
let zero_velocity (model : t) : velocity =
  { v_wxh = M.zeros (M.row_num model.wxh) (M.col_num model.wxh);
    v_whh = M.zeros (M.row_num model.whh) (M.col_num model.whh);
    v_bh  = M.zeros (M.row_num model.bh)  (M.col_num model.bh);
    v_why = M.zeros (M.row_num model.why) (M.col_num model.why);
    v_by  = M.zeros (M.row_num model.by)  (M.col_num model.by); }


(* [update model grads vel lr momentum] applies one SGD + momentum step:
     v  <-  momentum * v + grad
     parameters  <-  parameters - lr * v
   The velocity [v] acts as a low-pass filter on the gradient signal: noise
   averages out across steps while consistent gradient directions accumulate,
   letting the optimiser punch through the flat loss regions that stall
   vanilla SGD on long-range sequence tasks like LFSR prediction.
   Returns the updated model and the new velocity for the next step. *)
let update (model : t) (g : grads) (vel : velocity) (lr : float) (momentum : float)
    : t * velocity =
  (* Update each velocity: v <- momentum * v + grad *)
  let v_wxh = M.add (M.scalar_mul momentum vel.v_wxh) g.d_wxh in
  let v_whh = M.add (M.scalar_mul momentum vel.v_whh) g.d_whh in
  let v_bh  = M.add (M.scalar_mul momentum vel.v_bh)  g.d_bh  in
  let v_why = M.add (M.scalar_mul momentum vel.v_why) g.d_why in
  let v_by  = M.add (M.scalar_mul momentum vel.v_by)  g.d_by  in
  let new_vel = { v_wxh; v_whh; v_bh; v_why; v_by } in
  (* Apply the velocity as the effective update: θ <- θ - lr * v *)
  let new_model = { model with
    wxh = M.sub model.wxh (M.scalar_mul lr v_wxh);
    whh = M.sub model.whh (M.scalar_mul lr v_whh);
    bh  = M.sub model.bh  (M.scalar_mul lr v_bh);
    why = M.sub model.why (M.scalar_mul lr v_why);
    by  = M.sub model.by  (M.scalar_mul lr v_by); } in
  (new_model, new_vel)


(* [zero_hidden model] returns a zeroed initial hidden state — used at the
   start of each training sequence. *)
let zero_hidden (model : t) : M.mat =
  M.zeros model.hidden_size 1
