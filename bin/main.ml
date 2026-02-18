(* main.ml
   Entry point. Just kicks off the training loop with default hyperparameters.
   Tweak the labelled arguments here to experiment with different settings. *)

let () =
  Train.train
    ~seed:0xDEADBEEFL   (* LFSR initial state — change to get a different stream *)
    ~hidden_size:32      (* RNN memory capacity                                   *)
    ~seq_len:64          (* BPTT unroll length                                    *)
    ~n_steps:2000        (* total gradient updates                                *)
    ~lr:0.005            (* SGD learning rate                                     *)
    ~clip_val:5.0        (* gradient clipping threshold                           *)
    ()
