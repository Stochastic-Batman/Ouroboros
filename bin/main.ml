(* main.ml
   Entry point. Just kicks off the training loop with default hyperparameters.
   Tweak the labelled arguments here to experiment with different settings. *)

let () =
  Train.train
    ~seed:0x95L
    ~hidden_size:64
    ~seq_len:128
    ~n_steps:5000
    ~lr:0.005
    ~momentum:0.9
    ~clip_val:5.0
    ()
