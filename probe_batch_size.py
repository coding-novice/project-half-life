# probe_batch_size.py  — run via slurm on the A40
import json, torch
from model import SalukiModel

PARAMS = "example_params_fixed.json"
AMP = "none"          # set to "bf16" to measure the AMP slope (see Q2)
TARGET_FRAC = 0.90    # aim for 90% of 48 GB
p = json.load(open(PARAMS)); pm = p["model"]
dev = "cuda"
amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(AMP)
def peak_mem(bs, steps=2):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = SalukiModel(**pm).to(dev); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    for _ in range(steps):
        x = torch.randn(bs, pm["seq_length"], pm["seq_depth"], device=dev)
        y = torch.randn(bs, 1, device=dev)
        opt.zero_grad()
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss = torch.nn.functional.mse_loss(m(x, 0), y) + m.l2_loss()
        loss.backward(); opt.step()
    mem = torch.cuda.max_memory_allocated() / 1e9
    del m, opt; return mem
m8, m16 = peak_mem(8), peak_mem(16)
slope = (m16 - m8) / 8.0
overhead = m8 - slope * 8
budget = 48 * TARGET_FRAC
b_max = int((budget - overhead) / slope)
print(f"overhead={overhead:.2f}GB  per-sample={slope*1000:.1f}MB  -> est max batch ≈ {b_max}")
print("confirming...", peak_mem(b_max), "GB at b=", b_max)