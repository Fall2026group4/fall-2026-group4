"""OpenWebText x Pretrained SAEs analysis pipeline (owned by Dhruv).

See the project spec ("OpenWebText x Pretrained SAEs - AWS Analysis
Pipeline") for the full design. One module per concern:

- data.py           stream OWT/WikiText, tokenize, chunk to a fixed length
- saes.py           load each SAE type, map hook points
- fidelity.py        FVU, L0, spliced CE loss, loss recovered
- features.py        firing frequency, dead/dense latents
- steering_cost.py   KL divergence and CE change under added directions

`scripts/owt_0X_*.py` are thin runners over these - no analysis logic
lives in the scripts themselves.
"""
