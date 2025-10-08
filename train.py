import time
import json
import os
import shutil
from collections import defaultdict
import random

# --- GPU/Device Setup ---
try:
    import cupy as np
    np.cuda.runtime.getDeviceCount() # Check for GPU
    print("--- Using CuPy for GPU acceleration ---")
    IS_GPU = True
except (ImportError, np.cuda.runtime.CUDARuntimeError):
    import numpy as np
    print("--- CuPy not found or no GPU available, using NumPy for CPU ---")
    IS_GPU = False

# Import the custom modules we've built
from autograd import Tensor
from numpy_layers import Transformer
from training_utils import CrossEntropyLoss, Adam
from bpe_tokenizer import BpeTokenizer


def get_params(model):
    """ Helper function to collect all model parameters """
    params = []
    if isinstance(model, Transformer):
        modules = [
            model.encoder_embedding, model.decoder_embedding,
            *model.encoder_layers, *model.decoder_layers,
            model.fc
        ]
        for module in modules:
            params.extend(get_params(module))
    elif hasattr(model, 'params'):
        params.extend(model.params)
    return params

def zero_all_grads(model):
    """ Helper function to zero out all gradients in the model """
    for p in get_params(model):
        p.zero_grad()

def load_corpus(en_path, ta_path):
    """Loads parallel corpus from two files."""
    with open(en_path, 'r', encoding='utf-8') as f_en, \
         open(ta_path, 'r', encoding='utf-8') as f_ta:
        en_lines = [line.strip() for line in f_en.readlines()]
        ta_lines = [line.strip() for line in f_ta.readlines()]
    
    assert len(en_lines) == len(ta_lines), "Mismatched number of lines in corpus files!"
    corpus = list(zip(en_lines, ta_lines))
    print(f"Loaded {len(corpus)} sentence pairs.")
    return corpus

def preprocess_and_save_batches(corpus, tokenizer, batch_size, max_seq_len, save_dir="processed_batches"):
    """
    Pre-processes the entire corpus, creates padded batches, and saves them to disk.
    This avoids re-processing data every epoch, speeding up training significantly.
    """
    if os.path.exists(save_dir) and os.listdir(save_dir):
        response = input(f"Found pre-processed batches in '{save_dir}'. Re-process? (yes/no): ").lower()
        if response in ['n', 'no']:
            print("Skipping pre-processing. Using existing batches.")
            return sorted([os.path.join(save_dir, f) for f in os.listdir(save_dir) if f.endswith('.npz')])
        else:
            print("Re-processing data...")
            shutil.rmtree(save_dir)

    print(f"No pre-processed batches found or re-processing requested. Starting pre-processing...")
    os.makedirs(save_dir, exist_ok=True)

    processed_data = []
    print("Tokenizing data...")
    total_sentences = len(corpus)
    for i, (src_text, tgt_text) in enumerate(corpus):
        src_seq = tokenizer.encode(src_text)[:max_seq_len]
        tgt_seq = tokenizer.encode(tgt_text)[:max_seq_len - 1]
        tgt_input = [tokenizer.vocab.get('<sos>', 1)] + tgt_seq
        tgt_expected = tgt_seq + [tokenizer.vocab.get('<eos>', 2)]
        processed_data.append((src_seq, tgt_input, tgt_expected))
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total_sentences} sentences...", end='\r')

    processed_data.sort(key=lambda x: len(x[0]))
    print("Tokenized and sorted all data by length.")

    batch_files = []
    pad_id = tokenizer.vocab.get('<pad>', 0)
    import numpy as host_np

    print("Creating and saving batches...")
    num_batches = (len(processed_data) + batch_size - 1) // batch_size
    for i in range(num_batches):
        batch = processed_data[i*batch_size:(i+1)*batch_size]
        if not batch: continue
        src_seqs, tgt_inputs, tgt_expected = zip(*batch)
        max_src_len = max(len(s) for s in src_seqs)
        max_tgt_len = max(len(s) for s in tgt_inputs)
        src_padded = host_np.array([s + [pad_id] * (max_src_len - len(s)) for s in src_seqs])
        tgt_padded = host_np.array([s + [pad_id] * (max_tgt_len - len(s)) for s in tgt_inputs])
        expected_padded = host_np.array([s + [pad_id] * (max_tgt_len - len(s)) for s in tgt_expected])
        batch_filename = os.path.join(save_dir, f"batch_{i:05d}.npz")
        host_np.savez_compressed(batch_filename, src=src_padded, tgt=tgt_padded, expected=expected_padded)
        batch_files.append(batch_filename)
        if (i + 1) % 100 == 0:
            print(f"  Saved {i + 1}/{num_batches} batches...", end='\r')

    print(f"Saved {len(batch_files)} pre-processed batches to '{save_dir}'.")
    return sorted(batch_files)

def save_checkpoint(model, optimizer, epoch, batch_idx, filename="checkpoint.npz"):
    """Saves model and optimizer state."""
    params = get_params(model)
    # Use host numpy to save, as cupy arrays can't be pickled directly by savez.
    import numpy as host_np
    param_data = {f"param_{i}": np.asnumpy(p.data()) for i, p in enumerate(params)}
    optimizer_data = {
        "m": [host_np.array(m) for m in optimizer.m],
        "v": [host_np.array(v) for v in optimizer.v],
        "t": host_np.array(optimizer.t)
    }
    progress_data = {"epoch": host_np.array(epoch), "batch_idx": host_np.array(batch_idx)}
    
    host_np.savez(filename, **param_data, **optimizer_data, **progress_data)

def load_checkpoint(model, optimizer, filename="checkpoint.npz"):
    """Loads model and optimizer state."""
    data = np.load(filename, allow_pickle=True)
    params = get_params(model)
    for i, p in enumerate(params):
        p._data = np.asarray(data[f"param_{i}"])
    optimizer.m = [np.asarray(m) for m in data["m"]]
    optimizer.v = [np.asarray(v) for v in data["v"]]
    optimizer.t = int(data["t"])
    return int(data["epoch"]), int(data["batch_idx"]) + 1

if __name__ == "__main__":
    print("--- Starting Transformer Training ---")

    # 1. Hyperparameters
    D_MODEL = 48
    NUM_HEADS = 4
    NUM_LAYERS = 3
    D_FF = 96
    MAX_SEQ_LENGTH = 128 # Further reduced to decrease memory usage
    DROPOUT = 0.1
    LEARNING_RATE = 0.0005
    EPOCHS = 10
    BATCH_SIZE = 4 # Further reduced to decrease memory usage
    VOCAB_SIZE = 5000

    # 2. Load Data
    print("\nLoading training data...")
    train_corpus = load_corpus('data1-5.en', 'data1-5.ta')
    
    # 3. Train or Load Tokenizer
    tokenizer_path = "bpe_tokenizer.json"
    tokenizer = BpeTokenizer()
    try:
        tokenizer.load(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    except FileNotFoundError:
        print("\nTraining Tokenizer...")
        english_sentences = [pair[0] for pair in train_corpus]
        tamil_sentences = [pair[1] for pair in train_corpus]
        tokenizer.train(english_sentences + tamil_sentences, vocab_size=VOCAB_SIZE)
        tokenizer.save(tokenizer_path)
        print(f"Tokenizer trained and saved to {tokenizer_path}")

    SRC_VOCAB_SIZE = TGT_VOCAB_SIZE = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {SRC_VOCAB_SIZE}")

    # 4. Initialize Model, Loss, and Optimizer
    print("\nInitializing Model...")
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout_p=DROPOUT
    )
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(get_params(model), lr=LEARNING_RATE)
    print("Model, Loss, and Optimizer initialized.")

    # 5. Pre-process data and load/save checkpoints
    batch_files = preprocess_and_save_batches(train_corpus, tokenizer, BATCH_SIZE, MAX_SEQ_LENGTH)
    start_epoch, start_batch_idx = 0, 0
    checkpoint_path = "checkpoint.npz"
    if os.path.exists(checkpoint_path):
        print(f"--- Found checkpoint at '{checkpoint_path}'. Resuming training. ---")
        start_epoch, start_batch_idx = load_checkpoint(model, optimizer, checkpoint_path)

    # 6. The Training Loop
    print("\nStarting Training Loop...")
    start_time = time.time()
    total_batches = len(batch_files)
    checkpoint_interval = max(1, total_batches // 10) # Save ~10 times per epoch

    for epoch in range(start_epoch, EPOCHS):
        epoch_loss = 0
        shuffled_indices = list(range(total_batches))
        random.shuffle(shuffled_indices)

        # Skip batches already processed in a resumed epoch
        if start_batch_idx > 0:
            print(f"Resuming epoch {epoch} from batch {start_batch_idx}...")
        
        for i, batch_idx in enumerate(shuffled_indices):
            if i < start_batch_idx:
                continue
            
            batch_file = batch_files[batch_idx]
            with np.load(batch_file) as data:
                src_padded_cpu, tgt_padded_cpu, expected_padded_cpu = data['src'], data['tgt'], data['expected']
            
            # Move data to GPU if using CuPy
            src_padded = np.asarray(src_padded_cpu)
            tgt_padded = np.asarray(tgt_padded_cpu)
            expected_padded = np.asarray(expected_padded_cpu)

            # --- Forward Pass ---
            output = model.forward(Tensor(src_padded), Tensor(tgt_padded))

            # --- Calculate Loss ---
            output_reshaped = output.reshape(-1, SRC_VOCAB_SIZE)
            expected_reshaped = expected_padded.flatten()
            
            loss = loss_fn.forward(output_reshaped, expected_reshaped)
            current_batch_loss = loss.data()
            epoch_loss += current_batch_loss

            # --- Backward Pass ---
            zero_all_grads(model)
            loss.backward()

            # --- Update Weights ---
            optimizer.step()
            
            # --- Print Progress Indicator ---
            print(f"  Epoch {epoch + 1}/{EPOCHS} | Batch {i + 1}/{total_batches} | Loss: {current_batch_loss:.4f}   ", end='\r')
            
            # --- Save Checkpoint Periodically ---
            if (i + 1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, i, checkpoint_path)
                # The print below is commented out to avoid disrupting the progress bar
                # print(f"  ... Checkpoint saved at epoch {epoch+1}, batch {i+1}/{total_batches} ...")

        # Reset start_batch_idx for the next epoch
        start_batch_idx = 0
        avg_epoch_loss = epoch_loss / total_batches
        # Print a clean line for the final epoch summary
        print(f"Epoch {epoch + 1}/{EPOCHS} | Average Loss: {avg_epoch_loss:.4f}                                ")
        save_checkpoint(model, optimizer, epoch + 1, 0, checkpoint_path) # Save at end of epoch
    end_time = time.time()
    print(f"\nTraining finished in {end_time - start_time:.2f} seconds.")

    # 7. Evaluation on Test Set
    print("\n--- Evaluating on Test Set ---")
    test_corpus = load_corpus('data.en6', 'data.ta6')

    # Set all dropout layers in the model to evaluation mode
    for p in get_params(model):
        # This is a bit of a hack; a better way would be to have a proper
        # model.eval() method that traverses the module tree.
        if hasattr(p, 'eval'):
            p.eval()
    # Also set dropout on the main transformer class if it has one
    if hasattr(model, 'dropout') and hasattr(model.dropout, 'eval'):
        model.dropout.eval()


    def translate(sentence):
        src_seq = tokenizer.encode(sentence)[:MAX_SEQ_LENGTH]
        if not src_seq: return "<empty input>"
        
        # Ensure input is on the correct device (GPU/CPU)
        src_array = np.asarray([src_seq])
        src_tensor = Tensor(src_array)
        tgt_seq = [tokenizer.vocab.get('<sos>', 1)]
        
        for _ in range(MAX_SEQ_LENGTH):
            tgt_array = np.asarray([tgt_seq])
            tgt_tensor = Tensor(tgt_array)
            with np.errstate(all='ignore'):
                output = model.forward(src_tensor, tgt_tensor)
            
            next_token_id = np.argmax(output.data()[0, -1, :])
            if next_token_id == tokenizer.vocab.get('<eos>', 2) or len(tgt_seq) >= MAX_SEQ_LENGTH:
                break
            tgt_seq.append(next_token_id)
            
        return tokenizer.decode(tgt_seq[1:])

    for i in range(min(5, len(test_corpus))):
        src_sentence, tgt_sentence = test_corpus[i]
        translation = translate(src_sentence)
        print("-" * 20)
        print(f"SOURCE:    {src_sentence}")
        print(f"TARGET:    {tgt_sentence}")
        print(f"PREDICTED: {translation}")
