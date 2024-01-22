[The tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)

Requires an appropriate build of PyTorch. Tested using [rocm/docker](https://hub.docker.com/r/rocm/pytorch)

Training: `python run.py --train --context 128 --dimensions 256 --iterations 2500 --filename model_file_name`

Running: `python run.py --run --context 128 --dimensions 256 --output 300 --filename model_file_name`
