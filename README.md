# minit2i-tinygrad
tinygrad implementation of MiniT2I.

Refer to https://huggingface.co/MiniT2I/MiniT2I/blob/main/pipeline.py

Currently, it is significantly slower compared to the PyTorch (diffusers) implementation.

## Performance / Benchmark

Tested on an **RTX 4060**, minit2i-b16, 100 steps:

| Implementation | Model Loading | Inference Time |
| :--- | :--- | :--- |
| **PyTorch (diffusers)** | < 1 second | ~10 seconds |
| **tinygrad** | ~5 seconds | ~2 minutes |

<table>
  <tr>
    <td align="center"><img src="docs\minit2i-b16.png" width="400"><br><b>diffusers</b></td>
    <td align="center"><img src="docs\tiny.png" width="400"><br><b>tinygrad</b></td>
  </tr>
</table>
