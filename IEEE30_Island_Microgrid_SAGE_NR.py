# -*- coding: utf-8 -*-
"""
Standalone Microgrid OPF and Sizing Tool — IEEE 30-Bus (GraphSAGE + NR version)
=======================================================================
Mirror of IEEE30_Island_Microgrid_GAT_NR.py, but with the GraphSAGE
architecture (edge-encoder MLP + SAGEConv, no edge_dim argument).

Outputs:
  * design_grid_SAGE_only.csv   — SAGE design sweep (feas_prob per cell)
  * design_grid_SAGE_vs_NR.csv  — SAGE prob + NR feasibility + NR reason
                                  + per-channel NR violation counts
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, recall_score,
                             precision_score, f1_score)
from collections import defaultdict
import copy
import csv

# ============================================================
# 0. CONFIG  (identical to GAT version)
# ============================================================
FARM_SIZES_MW   = np.arange(0, 151, 10)                    # 16 values
BATT_E_MWH      = np.array([50, 100, 150, 200, 250, 300])  # 6 values
BATT_DURATION_H = 4.0
BATT_P_MAX      = BATT_E_MWH / BATT_DURATION_H
BATT_Q_MAX      = 0.5 * BATT_P_MAX

ROOFTOP_MW_EACH = 2.0
N_ROOFTOP       = 29
FARM_BUS        = 12
BATT_BUS        = 1
V_MIN, V_MAX    = 0.95, 1.05
SOC_MIN, SOC_MAX = 0.10, 0.90
N_SEEDS         = 20

LOAD_PROFILES  = {"night": 0.55, "morning": 0.80, "noon": 0.95,
                  "peak": 1.10, "low": 0.65}
SOLAR_PROFILES = {"night": 0.00, "morning": 0.40, "noon": 0.95,
                  "peak": 0.60, "low": 0.20}
BATT_P_PROFILES   = {"night": 30.0, "morning": 20.0, "noon": -30.0,
                     "peak": 40.0, "low": -10.0}
BATT_SOC_PROFILES = {"night": 0.70, "morning": 0.50, "noon": 0.30,
                     "peak": 0.60, "low": 0.40}

# SAGE hyperparameters
HIDDEN        = 64
N_LAYERS      = 6
EPOCHS        = 200
WARMUP_EPOCHS = 20
LR            = 3e-4
WEIGHT_DECAY  = 1e-5
LAMBDA_V      = 1.0
LAMBDA_TH     = 10.0
LAMBDA_VIOL   = 0.05
LAMBDA_FEAS   = 0.5
BATCH_SIZE    = 32
SAGE_AGGR     = "mean"

# ============================================================
# 1. ISLANDED NETWORK  (identical to GAT version)
# ============================================================
BASE_NET = pn.case30()
BASE_NET.ext_grid.drop(BASE_NET.ext_grid.index, inplace=True)
BASE_NET.ext_grid = BASE_NET.ext_grid.iloc[0:0].copy()
BASE_LOADS = BASE_NET.load[["p_mw", "q_mvar"]].copy()
BASE_GENS  = BASE_NET.gen[["p_mw", "vm_pu"]].copy()
BASE_BUSES = list(BASE_NET.bus.index)


def build_island_net(farm_mw, batt_e_mwh, load_scale, solar_scale,
                     batt_p_profile, batt_soc):
    net = copy.deepcopy(BASE_NET)
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    net.ext_grid = net.ext_grid.iloc[0:0].copy()

    pp.create_storage(
        net, bus=BATT_BUS, p_mw=batt_p_profile,
        max_e_mwh=float(batt_e_mwh),
        max_p_mw=float(batt_e_mwh / BATT_DURATION_H),
        min_p_mw=float(-batt_e_mwh / BATT_DURATION_H),
        soc_percent=float(batt_soc * 100),
        name="Grid-forming battery")
    pp.create_ext_grid(net, bus=BATT_BUS, vm_pu=1.00, va_degree=0.0,
                       name="GFM reference", in_service=True,
                       controllable=False)

    net.load.p_mw   *= load_scale
    net.load.q_mvar *= load_scale

    for bus_idx in net.bus.index:
        if bus_idx == BATT_BUS:
            continue
        pp.create_sgen(net, bus=bus_idx,
                       p_mw=ROOFTOP_MW_EACH * solar_scale,
                       q_mvar=0.0, name=f"Roof@{bus_idx}", type="solar")

    if farm_mw > 0:
        q = farm_mw * solar_scale * np.tan(np.arccos(0.95))
        pp.create_sgen(net, bus=FARM_BUS,
                       p_mw=farm_mw * solar_scale, q_mvar=q,
                       name=f"Farm{farm_mw:.0f}", type="solar")

    return (net, BATT_BUS,
            batt_e_mwh / BATT_DURATION_H,
            0.5 * batt_e_mwh / BATT_DURATION_H)


def solve_and_check_island(net, batt_bus, batt_p_max, batt_q_max,
                           batt_soc_init):
    """
    Returns: ok, reason, V, th, P, Q, soc, viol_mask
    viol_mask shape (n_bus, 3): [:,0]=voltage, [:,1]=SoC, [:,2]=P/Q
    """
    n_bus = len(net.bus)
    try:
        pp.runpp(net, algorithm="nr", init="flat", max_iteration=50,
                 tolerance_mva=1e-6, calculate_voltage_angles=True)
    except pp.LoadflowNotConverged:
        return (False, "NR fail", None, None, None, None, None,
                np.full((n_bus, 3), np.nan, dtype=np.float32))

    P  = net.res_ext_grid.p_mw.iloc[0]
    Q  = net.res_ext_grid.q_mvar.iloc[0]
    V  = net.res_bus.vm_pu.values
    th = np.radians(net.res_bus.va_degree.values)

    viol = np.zeros((n_bus, 3), dtype=np.float32)
    viol[:, 0] = ((V < V_MIN) | (V > V_MAX)).astype(np.float32)

    E   = batt_p_max * BATT_DURATION_H
    soc = batt_soc_init - P / E
    soc_viol = int(soc < SOC_MIN or soc > SOC_MAX)
    viol[batt_bus, 1] = float(soc_viol)

    pq_viol = int(abs(P) > batt_p_max + 1e-3 or abs(Q) > batt_q_max + 1e-3)
    viol[batt_bus, 2] = float(pq_viol)

    if abs(P) > batt_p_max + 1e-3:
        return False, f"P_batt {P:.1f}>{batt_p_max:.1f}", V, th, P, Q, None, viol
    if abs(Q) > batt_q_max + 1e-3:
        return False, f"Q_batt {Q:.1f}>{batt_q_max:.1f}", V, th, P, Q, None, viol
    if (V < V_MIN).any() or (V > V_MAX).any():
        return False, f"V viol ({V.min():.3f},{V.max():.3f})", V, th, P, Q, None, viol
    if soc < SOC_MIN or soc > SOC_MAX:
        return False, f"SoC {soc:.2f} out of band", V, th, P, Q, soc, viol

    return True, "OK", V, th, P, Q, soc, viol


# ============================================================
# 2. NETWORK -> PyG  (13 node features, 6 edge features, 3-ch viol)
# ============================================================
def net_to_pyg(net, V, theta, design, operating,
               feasible, P_batt, Q_batt, soc_final, viol_mask):
    n_bus = len(net.bus)

    Pd      = net.load.p_mw.reindex(net.bus.index, fill_value=0).values / net.sn_mva
    Qd      = net.load.q_mvar.reindex(net.bus.index, fill_value=0).values / net.sn_mva
    Pg_conv = net.gen.p_mw.reindex(net.bus.index, fill_value=0).values / net.sn_mva
    V_set   = net.gen.vm_pu.reindex(net.bus.index, fill_value=1.0).values

    Pg_ren = np.zeros(n_bus); Qg_ren = np.zeros(n_bus)
    for _, s in net.sgen.iterrows():
        Pg_ren[int(s.bus)] += s.p_mw / net.sn_mva
        Qg_ren[int(s.bus)] += s.q_mvar / net.sn_mva

    type_oh = np.zeros((n_bus, 3))
    slack_buses = {BATT_BUS}
    pv_buses    = set(net.gen.bus.values) - slack_buses
    for i in net.bus.index:
        if i in slack_buses: type_oh[i, 0] = 1.0
        elif i in pv_buses:  type_oh[i, 1] = 1.0
        else:                type_oh[i, 2] = 1.0

    farm_n  = design["farm_mw"] / 150.0
    batt_n  = design["batt_e_mwh"] / 300.0
    load_n  = operating["load"]
    globals_ = np.tile([farm_n, batt_n, load_n], (n_bus, 1))

    X = np.hstack([Pd[:, None], Qd[:, None], Pg_conv[:, None],
                   Pg_ren[:, None], Qg_ren[:, None], V_set[:, None],
                   type_oh, globals_]).astype(np.float32)
    X = torch.tensor(X, dtype=torch.float)

    edges, eattr = [], []
    for _, line in net.line.iterrows():
        i, j = int(line.from_bus), int(line.to_bus)
        bz = net.sn_mva / (net.bus.vn_kv.iloc[i] ** 2)
        r  = line.r_ohm_per_km * line.length_km * bz
        x  = line.x_ohm_per_km * line.length_km * bz
        b  = line.c_nf_per_km * line.length_km * 1e-9 * 2 * np.pi * 50 * bz
        y  = 1.0 / np.sqrt(r*r + x*x + 1e-12)
        edges.append([i, j]); eattr.append([r, x, b, y, 0.0, 0.0])

    for _, tr in net.trafo.iterrows():
        i, j = int(tr.hv_bus), int(tr.lv_bus)
        z = tr.vk_percent / 100.0
        r = z * (tr.vkr_percent / tr.vk_percent if tr.vk_percent > 0 else 0.0)
        x = np.sqrt(max(z*z - r*r, 0.0))
        y = 1.0 / np.sqrt(r*r + x*x + 1e-12)
        tap = (tr.tap_ratio if hasattr(tr, "tap_ratio") else 1.0) - 1.0
        edges.append([i, j]); eattr.append([r, x, 0.0, y, tap, 1.0])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr  = torch.tensor(eattr, dtype=torch.float)

    if V is not None:
        y_V    = torch.tensor(V, dtype=torch.float)
        y_th   = torch.tensor(theta, dtype=torch.float)
        y_viol = torch.tensor(viol_mask, dtype=torch.float)   # (n_bus, 3)
    else:
        y_V    = torch.full((n_bus,), float("nan"))
        y_th   = torch.full((n_bus,), float("nan"))
        y_viol = torch.full((n_bus, 3), float("nan"))

    return Data(
        x=X, edge_index=edge_index, edge_attr=edge_attr,
        y_V=y_V, y_th=y_th, y_viol=y_viol,
        y_feasible=torch.tensor([1.0 if feasible else 0.0], dtype=torch.float),
        farm_mw=torch.tensor([design["farm_mw"]], dtype=torch.float),
        batt_e_mwh=torch.tensor([design["batt_e_mwh"]], dtype=torch.float),
        P_batt=torch.tensor([P_batt if P_batt is not None else float("nan")],
                            dtype=torch.float),
        soc_final=torch.tensor([soc_final if soc_final is not None else float("nan")],
                               dtype=torch.float),
    )


# ============================================================
# 3. DATASET GENERATION
# ============================================================
def generate_island_dataset(n_seeds=N_SEEDS, seed=0):
    np.random.seed(seed)
    data_list, n_fail, n_infeas, n_total = [], 0, 0, 0
    feas_by_scn = defaultdict(lambda: [0, 0])

    for farm_mw in FARM_SIZES_MW:
        for batt_e in BATT_E_MWH:
            for ln, ls in LOAD_PROFILES.items():
                for sn, ss in SOLAR_PROFILES.items():
                    bp = BATT_P_PROFILES[ln] * (batt_e / 300.0)
                    for _ in range(n_seeds):
                        n_total += 1
                        j = np.random.uniform(0.95, 1.05)
                        net, bbus, pmax, qmax = build_island_net(
                            farm_mw, batt_e, ls * j, ss, bp,
                            BATT_SOC_PROFILES[ln])
                        ok, reason, V, th, Pb, Qb, soc, viol_mask = \
                            solve_and_check_island(
                                net, bbus, pmax, qmax, BATT_SOC_PROFILES[ln])
                        if not ok and V is None:
                            n_fail += 1
                        elif not ok:
                            n_infeas += 1
                        feas_by_scn[(round(ls, 2), round(ss, 2))][0] += int(ok)
                        feas_by_scn[(round(ls, 2), round(ss, 2))][1] += 1
                        data_list.append(net_to_pyg(
                            net, V, th,
                            design={"farm_mw": farm_mw, "batt_e_mwh": batt_e},
                            operating={"load": ls, "solar": ss},
                            feasible=ok, P_batt=Pb, Q_batt=Qb,
                            soc_final=soc, viol_mask=viol_mask))

    print(f"Generated {len(data_list)} samples "
          f"(NR fail={n_fail}, infeasible={n_infeas}, total={n_total})")

    all_viol = torch.cat(
        [d.y_viol[~torch.isnan(d.y_viol[:, 0])] for d in data_list], dim=0)
    n_valid = all_viol.shape[0]
    for ch, name in enumerate(["voltage", "SoC    ", "P/Q    "]):
        n_pos = int((all_viol[:, ch] == 1).sum().item())
        print(f"    violation channel {ch} ({name}): "
              f"{n_pos}/{n_valid} positives "
              f"({100*n_pos/max(n_valid,1):.2f}%)")

    print("\n  Feasibility rate by (load, solar):")
    for (ls, ss), (n_ok, n) in sorted(feas_by_scn.items()):
        print(f"    load={ls:.2f}  solar={ss:.2f} : "
              f"{n_ok:5d}/{n:<5d} ({100*n_ok/max(n,1):5.1f}%)")
    return data_list


# ============================================================
# 4. SAGE MODEL  (edge-encoder MLP + SAGEConv, 3-channel viol head)
# ============================================================
class StandaloneMicrogridSAGE(nn.Module):
    """
    GraphSAGE with an explicit edge-encoder MLP:
      1. Project each edge's 6-dim attribute to `hidden` dims.
      2. Scatter-add these projected edge features onto their source nodes.
      3. Add the result to the node encoder output before SAGE layers.
    SAGEConv does not accept edge_attr, hence the edge-encoder is needed.
    """
    def __init__(self, in_dim, edge_dim=6, hidden=64, n_layers=6,
                 aggr="mean"):
        super().__init__()
        self.in_dim, self.edge_dim = in_dim, edge_dim
        self.aggr = aggr

        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden))

        self.convs = nn.ModuleList([
            SAGEConv(hidden, hidden, aggr=aggr)
            for _ in range(n_layers)])
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(n_layers)])

        self.head_V = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.head_th = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.head_viol = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3))
        self.head_feas = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, data):
        x, ei = data.x, data.edge_index
        assert x.shape[1] == self.in_dim, \
            f"Expected {self.in_dim} node features, got {x.shape[1]}"

        ea = data.edge_attr
        if ea.dim() == 1:
            ea = ea.unsqueeze(-1)
        assert ea.shape[1] == self.edge_dim, \
            f"Expected {self.edge_dim} edge features, got {ea.shape[1]}"

        if getattr(data, "batch", None) is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            batch = data.batch

        h = self.encoder(x)

        # Edge-feature injection via scatter-add onto source nodes
        edge_feat = self.edge_encoder(ea)            # (n_edges, hidden)
        h_edge = torch.zeros(x.shape[0], h.shape[1], device=x.device)
        h_edge.index_add_(0, ei[0], edge_feat)
        h = h + h_edge

        # SAGE message passing (no edge_attr argument)
        for conv, norm in zip(self.convs, self.norms):
            h_res = h
            h = conv(h, ei)
            h = norm(F.relu(h)) + h_res

        h_mean = global_mean_pool(h, batch)
        h_max  = global_max_pool(h, batch)
        feas_logit = self.head_feas(
            torch.cat([h_mean, h_max], dim=-1)).squeeze(-1)

        return {
            "V":        self.head_V(h).squeeze(-1),
            "th":       self.head_th(h).squeeze(-1),
            "viol":     self.head_viol(h),           # (n_nodes, 3)
            "feasible": feas_logit,
        }


# ============================================================
# 5. TRAINING  (per-channel pos_weight, 3-channel BCE)
# ============================================================
def train_island_sage(model, train_loader, val_loader, device,
                      epochs=EPOCHS, warmup=WARMUP_EPOCHS, lr=LR):
    yv = torch.cat([b.y_viol[~torch.isnan(b.y_viol[:, 0])]
                    for b in train_loader], dim=0)
    pos_weight_viol = []
    for ch in range(3):
        n_pos = (yv[:, ch] == 1).sum().item()
        n_neg = (yv[:, ch] == 0).sum().item()
        pos_weight_viol.append(max(n_neg / max(n_pos, 1), 1.0))
    pos_weight_viol = torch.tensor(pos_weight_viol,
                                   dtype=torch.float32, device=device)
    print(f"pos_weight(viol) per channel [V, SoC, P/Q] = "
          f"[{pos_weight_viol[0]:.1f}, {pos_weight_viol[1]:.1f}, "
          f"{pos_weight_viol[2]:.1f}]")

    yf = torch.cat([b.y_feasible for b in train_loader])
    f_pos = (yf == 1).sum().item(); f_neg = (yf == 0).sum().item()
    feas_weight = torch.tensor(max(f_neg / max(f_pos, 1), 1.0),
                               dtype=torch.float32, device=device)
    print(f"pos_weight(feas)={feas_weight.item():.2f}")

    if warmup > 0:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.encoder.parameters():
            p.requires_grad = True
        for p in model.edge_encoder.parameters():
            p.requires_grad = True
        for p in model.convs[0].parameters():
            p.requires_grad = True
        for p in model.head_V.parameters():
            p.requires_grad = True

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=10, min_lr=1e-7)

    best, best_state = float("inf"), None
    switched = False

    for epoch in range(1, epochs + 1):
        if warmup > 0 and epoch == warmup + 1 and not switched:
            print("\n  >>> Phase 2 — unfreezing all parameters")
            for p in model.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(model.parameters(), lr=lr,
                                   weight_decay=WEIGHT_DECAY)
            switched = True

        model.train()
        tr_V = tr_th = tr_v = tr_f = 0.0
        for b in train_loader:
            b = b.to(device)
            opt.zero_grad()
            o = model(b)

            V_set = b.x[:, 5]
            slack = b.x[:, 6] > 0.5
            pv    = b.x[:, 7] > 0.5
            mV    = ~torch.isnan(b.y_V)
            mT    = ~torch.isnan(b.y_th)
            mV2   = ~torch.isnan(b.y_viol[:, 0])

            V_used  = torch.where(slack | pv, V_set, o["V"])
            th_used = torch.where(slack, torch.zeros_like(o["th"]), o["th"])

            lV = F.mse_loss(V_used[mV], b.y_V[mV]) if mV.any() \
                 else torch.tensor(0.0, device=device)
            lt = F.mse_loss(th_used[mT], b.y_th[mT]) if mT.any() \
                 else torch.tensor(0.0, device=device)
            lv = F.binary_cross_entropy_with_logits(
                o["viol"][mV2], b.y_viol[mV2],
                pos_weight=pos_weight_viol) if mV2.any() \
                 else torch.tensor(0.0, device=device)
            lf = F.binary_cross_entropy_with_logits(
                o["feasible"], b.y_feasible, pos_weight=feas_weight)

            loss = LAMBDA_V*lV + LAMBDA_TH*lt + LAMBDA_VIOL*lv + LAMBDA_FEAS*lf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_V  += lV.item() * b.num_graphs
            tr_th += lt.item() * b.num_graphs
            tr_v  += lv.item() * b.num_graphs
            tr_f  += lf.item() * b.num_graphs

        n = len(train_loader.dataset)
        tr_V, tr_th, tr_v, tr_f = tr_V/n, tr_th/n, tr_v/n, tr_f/n

        model.eval()
        va_maeV, va_feas_acc, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for b in val_loader:
                b = b.to(device)
                o = model(b)
                V_set = b.x[:, 5]
                slack = b.x[:, 6] > 0.5
                pv    = b.x[:, 7] > 0.5
                V_used = torch.where(slack | pv, V_set, o["V"])
                m = ~torch.isnan(b.y_V)
                if m.any():
                    va_maeV += (V_used[m] - b.y_V[m]).abs().sum().item()
                pred_f = (torch.sigmoid(o["feasible"]) > 0.5).float()
                va_feas_acc += (pred_f == b.y_feasible).sum().item()
                va_n += b.num_graphs

        va_maeV /= max(va_n, 1)
        va_feas_acc /= max(va_n, 1)
        sched.step(va_maeV)

        if va_maeV < best:
            best = va_maeV
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}  "
                  f"tr[V={tr_V:.2e} θ={tr_th:.2e} viol={tr_v:.2e} feas={tr_f:.2e}]  "
                  f"| val MAE_V={va_maeV:.5f}  feas_acc={va_feas_acc*100:.1f}%  "
                  f"lr={opt.param_groups[0]['lr']:.1e}")

    model.load_state_dict(best_state)
    return model


# ============================================================
# 6. EVALUATION  (per-channel violation metrics)
# ============================================================
def evaluate_island_sage(model, test_loader, device):
    model.eval()
    Vp, Vt, tp, tt, lv, vt, lf, ft = [], [], [], [], [], [], [], []

    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            o = model(b)
            V_set = b.x[:, 5]
            slack = b.x[:, 6] > 0.5
            pv    = b.x[:, 7] > 0.5
            mV = ~torch.isnan(b.y_V)
            mT = ~torch.isnan(b.y_th)
            mv = ~torch.isnan(b.y_viol[:, 0])

            Vp.append(torch.where(slack | pv, V_set, o["V"])[mV].cpu())
            Vt.append(b.y_V[mV].cpu())
            tp.append(torch.where(slack, torch.zeros_like(o["th"]), o["th"])[mT].cpu())
            tt.append(b.y_th[mT].cpu())
            lv.append(o["viol"][mv].cpu())
            vt.append(b.y_viol[mv].cpu())
            lf.append(o["feasible"].cpu())
            ft.append(b.y_feasible.cpu())

    Vp = torch.cat(Vp).numpy(); Vt = torch.cat(Vt).numpy()
    tp = torch.cat(tp).numpy(); tt = torch.cat(tt).numpy()
    lv = torch.cat(lv).numpy(); vt = torch.cat(vt).numpy().astype(int)
    lf = torch.cat(lf).numpy(); ft = torch.cat(ft).numpy().astype(int)

    mae_V  = np.abs(Vp - Vt).mean()
    rmse_V = np.sqrt(((Vp - Vt) ** 2).mean())
    mae_th = np.abs(tp - tt).mean()
    pf_p = (1/(1+np.exp(-lf)) > 0.5).astype(int)

    print("\n" + "="*62)
    print("TEST EVALUATION — SAGE Standalone Microgrid (3-channel viol)")
    print("="*62)
    print(f"MAE V       : {mae_V:.5f} pu")
    print(f"RMSE V      : {rmse_V:.5f} pu")
    print(f"MAE θ       : {mae_th:.5f} rad")

    viol_metrics = {}
    for ch, name in enumerate(["Voltage", "SoC    ", "P/Q    "]):
        pv_ch = (1/(1+np.exp(-lv[:, ch])) > 0.5).astype(int)
        acc  = accuracy_score(vt[:, ch], pv_ch)
        rec  = recall_score(vt[:, ch], pv_ch, zero_division=0)
        prec = precision_score(vt[:, ch], pv_ch, zero_division=0)
        viol_metrics[name.strip()] = (acc, rec, prec)
        print(f"Violation[{name}]  acc={acc*100:6.2f}%  "
              f"recall={rec*100:6.2f}%  precision={prec*100:6.2f}%  "
              f"(pos={int(vt[:, ch].sum())}, pred={int(pv_ch.sum())})")

    print(f"\nFeasibility acc  : {accuracy_score(ft, pf_p)*100:.2f} %")
    print(f"Feasibility prec : {precision_score(ft, pf_p, zero_division=0)*100:.2f} %")
    print(f"Feasibility rec  : {recall_score(ft, pf_p, zero_division=0)*100:.2f} %")

    return {"mae_V": mae_V, "mae_th": mae_th,
            "viol_metrics": viol_metrics,
            "feas_acc": accuracy_score(ft, pf_p),
            "feas_recall": recall_score(ft, pf_p, zero_division=0)}


# ============================================================
# 7. SAGE DESIGN SWEEP
# ============================================================
SAFETY_MARGIN = 0.01

def pareto_design_sweep(model, base_data, scenarios,
                        farm_sizes, batt_e_list, device, verbose=False):
    model.eval()
    results = []

    with torch.no_grad():
        for ls, ss in scenarios:
            if verbose:
                print(f"\n  ── Scenario  load={ls:.2f}  solar={ss:.2f} ──")
            full_grid = []
            for farm in farm_sizes:
                for batt in batt_e_list:
                    feat = base_data.x.clone().to(device)
                    feat[:, 9]  = farm / 150.0
                    feat[:, 10] = batt / 300.0
                    feat[:, 11] = ls

                    d = Data(x=feat,
                             edge_index=base_data.edge_index.to(device),
                             edge_attr=base_data.edge_attr.to(device))
                    o = model(d)

                    fp = float(torch.sigmoid(o["feasible"]).item())
                    V_set = feat[:, 5]
                    slack = feat[:, 6] > 0.5
                    pv    = feat[:, 7] > 0.5
                    V_used = torch.where(slack | pv, V_set, o["V"])

                    viol_pred = (torch.sigmoid(o["viol"]) > 0.5).int()
                    nv_V   = int(viol_pred[:, 0].sum().item())
                    nv_SoC = int(viol_pred[:, 1].sum().item())
                    nv_PQ  = int(viol_pred[:, 2].sum().item())

                    min_V   = float(V_used.min().item())
                    max_V   = float(V_used.max().item())
                    min_bus = int(V_used.argmin().item()) + 1
                    max_bus = int(V_used.argmax().item()) + 1
                    mL      = min_V - V_MIN
                    mH      = V_MAX - max_V

                    row = {
                        "farm_mw":     float(farm),
                        "batt_e_mwh":  float(batt),
                        "feas_prob":   fp,
                        "n_viol_V":    nv_V,
                        "n_viol_SoC":  nv_SoC,
                        "n_viol_PQ":   nv_PQ,
                        "min_V":       min_V,
                        "min_bus":     min_bus,
                        "margin_low":  mL,
                        "max_V":       max_V,
                        "max_bus":     max_bus,
                        "margin_high": mH,
                    }
                    full_grid.append(row)

            cand_sorted = sorted(
                full_grid, key=lambda c: (c["farm_mw"] + 0.2 * c["batt_e_mwh"]))
            pareto = []
            for c in cand_sorted:
                if c["feas_prob"] < 0.5:               continue
                if c["margin_low"] < SAFETY_MARGIN:    continue
                if c["margin_high"] < SAFETY_MARGIN:   continue
                dominated = any(
                    p["farm_mw"] <= c["farm_mw"]
                    and p["batt_e_mwh"] <= c["batt_e_mwh"]
                    and (p["farm_mw"] < c["farm_mw"]
                         or p["batt_e_mwh"] < c["batt_e_mwh"])
                    for p in pareto)
                if not dominated:
                    pareto.append(c)

            results.append({
                "scenario":  {"load": ls, "solar": ss},
                "full_grid": full_grid,
                "pareto":    pareto,
            })
    return results


# ============================================================
# 8. NR DESIGN SWEEP  (ground truth + SAGE-vs-NR agreement)
# ============================================================
def nr_design_sweep(scenarios, farm_sizes, batt_e_list):
    results = []
    for ls, ss in scenarios:
        ln = next(k for k, v in LOAD_PROFILES.items() if abs(v - ls) < 1e-6)
        grid = []
        for farm in farm_sizes:
            for batt in batt_e_list:
                bp = BATT_P_PROFILES[ln] * (batt / 300.0)
                net, bbus, pmax, qmax = build_island_net(
                    farm, batt, ls, ss, bp, BATT_SOC_PROFILES[ln])
                ok, reason, V, th, Pb, Qb, soc, viol_mask = \
                    solve_and_check_island(net, bbus, pmax, qmax,
                                           BATT_SOC_PROFILES[ln])
                grid.append({
                    "farm_mw": float(farm),
                    "batt_e_mwh": float(batt),
                    "feasible_NR": int(ok),
                    "reason": reason,
                    "n_viol_V":   int(np.nansum(viol_mask[:, 0])),
                    "n_viol_SoC": int(np.nansum(viol_mask[:, 1])),
                    "n_viol_PQ":  int(np.nansum(viol_mask[:, 2])),
                })
        feas = [g for g in grid if g["feasible_NR"] == 1]
        results.append({
            "load": ls, "solar": ss,
            "min_farm_NR": min((g["farm_mw"]    for g in feas), default=None),
            "min_batt_NR": min((g["batt_e_mwh"] for g in feas), default=None),
            "grid": grid,
        })
    return results


# ============================================================
# 9. MAIN
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1] Generating islanded dataset...")
    all_data = generate_island_dataset(n_seeds=N_SEEDS)
    in_dim   = all_data[0].x.shape[1]
    edge_dim = all_data[0].edge_attr.shape[1]
    print(f"    Node features: {in_dim}  (expect 13)")
    print(f"    Edge features: {edge_dim}")

    farm_lbl = [int(d.farm_mw.item()) for d in all_data]
    train, test = train_test_split(all_data, test_size=0.30,
                                   random_state=42, stratify=farm_lbl)
    val, test = train_test_split(test, test_size=0.50, random_state=42,
                                 stratify=[int(d.farm_mw.item()) for d in test])
    print(f"    Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test,  batch_size=BATCH_SIZE, shuffle=False)

    print("\n[2] Building StandaloneMicrogridSAGE...")
    model = StandaloneMicrogridSAGE(in_dim=in_dim, edge_dim=edge_dim,
                                    hidden=HIDDEN, n_layers=N_LAYERS,
                                    aggr=SAGE_AGGR).to(device)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    b = next(iter(train_loader)).to(device)
    with torch.no_grad():
        o = model(b)
    print(f"    Smoke: V{o['V'].shape} θ{o['th'].shape} "
          f"viol{o['viol'].shape} feas{o['feasible'].shape}")
    assert o["feasible"].shape == b.y_feasible.shape
    assert o["viol"].shape[1] == 3

    print("\n[3] Training...")
    model = train_island_sage(model, train_loader, val_loader, device)
    torch.save(model.state_dict(), "standalone_microgrid_sage.pt")
    print("    Saved → standalone_microgrid_sage.pt")

    print("\n[4] Evaluating on test set...")
    metrics = evaluate_island_sage(model, test_loader, device)

    print("\n" + "=" * 70)
    print("[5] DESIGN SWEEP — full grid per scenario (SAGE)")
    print("=" * 70)

    base = train[0]
    all_scenarios = [(ls, ss) for ls in sorted(set(LOAD_PROFILES.values()))
                                for ss in sorted(set(SOLAR_PROFILES.values()))]

    sweep_results = pareto_design_sweep(
        model, base,
        scenarios=all_scenarios,
        farm_sizes=FARM_SIZES_MW,
        batt_e_list=BATT_E_MWH,
        device=device, verbose=False)

    csv_path = "design_grid_SAGE_only.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "load", "solar", "farm_mw", "batt_e_mwh",
            "feas_prob", "n_viol_V", "n_viol_SoC", "n_viol_PQ",
            "min_V", "min_bus", "margin_low",
            "max_V", "max_bus", "margin_high",
        ])
        for r in sweep_results:
            for row in r["full_grid"]:
                writer.writerow([
                    r["scenario"]["load"], r["scenario"]["solar"],
                    row["farm_mw"], row["batt_e_mwh"],
                    f"{row['feas_prob']:.4f}",
                    row["n_viol_V"], row["n_viol_SoC"], row["n_viol_PQ"],
                    f"{row['min_V']:.6f}", row["min_bus"],
                    f"{row['margin_low']:+.6f}",
                    f"{row['max_V']:.6f}", row["max_bus"],
                    f"{row['margin_high']:+.6f}",
                ])
    print(f"\n[6] SAGE-only design grid saved → {csv_path}")

    print("\n" + "=" * 70)
    print("[7] NR DESIGN SWEEP — ground truth comparison")
    print("=" * 70)

    nr_results = nr_design_sweep(all_scenarios, FARM_SIZES_MW, BATT_E_MWH)

    csv_path = "design_grid_SAGE_vs_NR.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["load", "solar", "farm_mw", "batt_e_mwh",
                    "feas_prob_SAGE", "feasible_NR", "nr_reason",
                    "n_viol_V", "n_viol_SoC", "n_viol_PQ"])
        for r_nr, r_sg in zip(nr_results, sweep_results):
            for g_nr in r_nr["grid"]:
                g_sg = next(c for c in r_sg["full_grid"]
                            if abs(c["farm_mw"]    - g_nr["farm_mw"])    < 1e-6
                            and abs(c["batt_e_mwh"] - g_nr["batt_e_mwh"]) < 1e-6)
                w.writerow([
                    r_nr["load"], r_nr["solar"],
                    g_nr["farm_mw"], g_nr["batt_e_mwh"],
                    f"{g_sg['feas_prob']:.4f}",
                    g_nr["feasible_NR"], g_nr["reason"],
                    g_nr["n_viol_V"], g_nr["n_viol_SoC"], g_nr["n_viol_PQ"],
                ])
    print(f"  Saved → {csv_path}")

    print("\n  load  solar | NR min (farm, batt) | SAGE min (farm, batt) | agree?")
    print("  " + "-" * 76)
    n_agree = 0
    for r_nr, r_sg in zip(nr_results, sweep_results):
        sg_feas = [c for c in r_sg["full_grid"] if c["feas_prob"] >= 0.5]
        sg_min_farm = min((c["farm_mw"]    for c in sg_feas), default=None)
        sg_min_batt = min((c["batt_e_mwh"] for c in sg_feas), default=None)
        agree = (r_nr["min_farm_NR"] == sg_min_farm
                 and r_nr["min_batt_NR"] == sg_min_batt)
        n_agree += int(agree)
        print(f"  {r_nr['load']:.2f}  {r_nr['solar']:.2f} | "
              f"({r_nr['min_farm_NR']}, {r_nr['min_batt_NR']}) | "
              f"({sg_min_farm}, {sg_min_batt}) | "
              f"{'YES' if agree else 'NO'}")
    print(f"\n  Agreement: {n_agree}/{len(nr_results)} scenarios")

    print("\n" + "=" * 70)
    print("[8] SAGE FINAL SUMMARY")
    print("=" * 70)
    print(f"  MAE V             : {metrics['mae_V']:.5f} pu")
    print(f"  MAE θ             : {metrics['mae_th']:.5f} rad")
    for name, (acc, rec, prec) in metrics["viol_metrics"].items():
        print(f"  Violation[{name}] : acc={acc*100:.2f}%  "
              f"recall={rec*100:.2f}%  precision={prec*100:.2f}%")
    print(f"  Feasibility acc   : {metrics['feas_acc']*100:.2f} %")
    print(f"  Feasibility recall: {metrics['feas_recall']*100:.2f} %")

    return model, metrics, sweep_results, nr_results


if __name__ == "__main__":
    model, metrics, sweep_results, nr_results = main()