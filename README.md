[The tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)

Requires an appropriate build of PyTorch. Tested using [rocm/pytorch](https://hub.docker.com/r/rocm/pytorch)

Training: `python run_model.py --train --blocks 6 --context 128 --dimensions 256 --iterations 2500 --filename model_file_name`

Running: `python run_model.py --run --blocks 6 --context 128 --dimensions 256 --output 300 --filename model_file_name`

(Optional) Training the BPE tokenizer: `python run_tokenizer.py -v 512 -i input.txt -o tok_512`

(Optional) Using the BPE tokenizer: `python run_model.py ... --tokenizer tok_512`
