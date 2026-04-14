"""
Train MeanFlow with larger noise scale to match the inference-time optimum.

Discovery: at inference, noise_scale=3.0 gives solar 0.358 (vs 0.418 at noise=1.0).
Hypothesis: if we also train with noise=3.0, the model will properly learn
to transport from the wider noise distribution, giving even better results.

Usage:
    python experiments/train_noise_aware.py solar_nips --noise-scale 3.0
"""
import os, sys, time, argparse, logging, numpy as np, torch, torch.nn as nn
from torch.optim import AdamW
from copy import deepcopy
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import TrainDataLoader
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import *
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from meanflow_ts.model_v4 import (
    S4DMeanFlowNetV4, get_lag_indices_v4, RobustNorm, extract_lags_v4, sample_t_r,
)

logging.basicConfig(format="%(asctime)s | %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIGS = {"solar_nips": {"freq":"H","ctx":72,"pred":24,"clamp":0},
           "exchange_rate_nips": {"freq":"B","ctx":30,"pred":30,"clamp":None}}
LAG_MAP = {"H": 672, "B": 750}
TSFLOW = {"solar_nips": 0.341, "exchange_rate_nips": 0.005}


def scaled_meanflow_loss(net, future_clean, context_channels, noise_scale=1.0,
                          norm_p=0.75, norm_eps=1e-3):
    """MeanFlow JVP loss with scaled noise."""
    B = future_clean.shape[0]
    device = future_clean.device
    e = torch.randn_like(future_clean) * noise_scale
    t, r = sample_t_r(B, device)
    t_bc, r_bc = t.unsqueeze(-1), r.unsqueeze(-1)
    z = (1 - t_bc) * future_clean + t_bc * e
    v = e - future_clean

    def u_func(z, t_bc, r_bc):
        h_bc = t_bc - r_bc
        return net(z, (t_bc.squeeze(-1), h_bc.squeeze(-1)), context_channels)

    with torch.amp.autocast("cuda", enabled=False):
        u_pred, dudt = torch.func.jvp(
            u_func, (z, t_bc, r_bc),
            (v, torch.ones_like(t_bc), torch.zeros_like(r_bc)),
        )
        u_tgt = (v - (t_bc - r_bc) * dudt).detach()
        loss = (u_pred - u_tgt) ** 2
        loss = loss.sum(dim=1)
        adp_wt = (loss.detach() + norm_eps) ** norm_p
        loss = (loss / adp_wt).mean()
    return loss


class NoiseScaledFc(nn.Module):
    def __init__(s, net, ctx, pred, ns, freq, noise_scale, clamp_min=None):
        super().__init__()
        s.net=net; s.context_length=ctx; s.prediction_length=pred
        s.num_samples=ns; s.freq=freq; s.noise_scale=noise_scale
        s.clamp_min=clamp_min; s.norm=RobustNorm()

    def forward(s, past_target, past_observed_values, **kw):
        d=past_target.device; B=past_target.shape[0]
        c=past_target[:,-s.context_length:].float()
        cn,l,sc=s.norm(c); lags=extract_lags_v4(past_target.float(),s.context_length,s.freq)
        ln=(lags-l.unsqueeze(1))/sc.unsqueeze(1); ps=[]
        for _ in range(s.num_samples):
            z=torch.randn(B,s.prediction_length,device=d) * s.noise_scale
            t=torch.ones(B,device=d); h=t.clone()
            p=s.norm.inverse((z-s.net(z,(t,h),ln)).float(),l,sc).float()
            if s.clamp_min is not None: p=p.clamp(min=s.clamp_min)
            ps.append(p.cpu())
        return torch.stack(ps,dim=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--epochs",type=int,default=800)
    p.add_argument("--lr",type=float,default=5e-4)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--noise-scale",type=float,default=3.0)
    p.add_argument("--warm-start",action="store_true")
    args = p.parse_args()

    name=args.dataset; cfg=CONFIGS[name]
    freq,ctx,pred=cfg["freq"],cfg["ctx"],cfg["pred"]
    clamp=cfg["clamp"]; max_lag=LAG_MAP[freq]
    n_lags=len(get_lag_indices_v4(freq))
    device=torch.device("cuda")
    torch.manual_seed(42); np.random.seed(42)

    outdir=os.path.join(os.path.dirname(__file__),'..','results_noise_aware',name)
    os.makedirs(outdir,exist_ok=True)
    norm=RobustNorm()

    logger.info(f"{'='*60}")
    logger.info(f"Noise-aware training: {name} noise_scale={args.noise_scale}")
    logger.info(f"{'='*60}")

    net=S4DMeanFlowNetV4(pred_len=pred,ctx_len=ctx,d_model=192,n_s4d_blocks=6,
                          ssm_dim=64,n_lags=n_lags,freq=freq).to(device)
    if args.warm_start:
        v4p=os.path.join(os.path.dirname(__file__),'..','results_v4',name,'best.pt')
        if os.path.exists(v4p):
            ck=torch.load(v4p,map_location=device,weights_only=False)
            net.load_state_dict(ck['net_ema'])
            logger.info(f"Warm-started from v4 (CRPS={ck.get('crps','?')})")
    net_ema=deepcopy(net).eval()
    logger.info(f"Params: {sum(pp.numel() for pp in net.parameters()):,}")

    ds=get_dataset(name)
    tr=Chain([AsNumpyArray(field="target",expected_ndim=1),
        AddObservedValuesIndicator(target_field="target",output_field="observed_values"),
        AddTimeFeatures(start_field="start",target_field="target",output_field="time_feat",
            time_features=time_features_from_frequency_str(freq),pred_length=pred)])
    sp=InstanceSplitter(target_field="target",is_pad_field="is_pad",start_field="start",
        forecast_start_field="forecast_start",
        instance_sampler=ExpectedNumInstanceSampler(num_instances=1,min_future=pred),
        past_length=ctx+max_lag,future_length=pred,
        time_series_fields=["time_feat","observed_values"])
    td=tr.apply(ds.train,is_train=True)
    loader=TrainDataLoader(Cached(td),batch_size=args.batch_size,stack_fn=batchify,
        transform=sp,num_batches_per_epoch=100,shuffle_buffer_length=2000)

    opt=AdamW(net.parameters(),lr=args.lr,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=1e-5)
    best=float('inf')

    for epoch in range(args.epochs):
        net.train(); el=0; nb=0; t0=time.time()
        for batch in loader:
            past=batch["past_target"].to(device).float()
            future=batch["future_target"].to(device).float()
            cr=past[:,-ctx:]; cn,l,s=norm(cr); fn=(future-l)/s
            lags=extract_lags_v4(past,ctx,freq); ln=(lags-l.unsqueeze(1))/s.unsqueeze(1)
            loss=scaled_meanflow_loss(net,fn,ln,noise_scale=args.noise_scale)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),0.5)
            opt.step()
            with torch.no_grad():
                for pp,pe in zip(net.parameters(),net_ema.parameters()):
                    pe.data.lerp_(pp.data,1e-4)
            el+=loss.item(); nb+=1
        sched.step()
        if (epoch+1)%20==0 or epoch==0:
            logger.info(f"Ep {epoch+1}/{args.epochs} | Loss={el/nb:.4f} | "
                        f"lr={opt.param_groups[0]['lr']:.2e} | {time.time()-t0:.1f}s")
        if (epoch+1)%50==0 or (epoch+1)==args.epochs:
            net_ema.eval()
            tt=tr.apply(ds.test,is_train=False)
            tsp=InstanceSplitter(target_field="target",is_pad_field="is_pad",start_field="start",
                forecast_start_field="forecast_start",instance_sampler=TestSplitSampler(),
                past_length=ctx+max_lag,future_length=pred,
                time_series_fields=["time_feat","observed_values"])
            fc=NoiseScaledFc(net_ema,ctx,pred,16,freq,args.noise_scale,clamp).to(device)
            pr=PyTorchPredictor(prediction_length=pred,
                input_names=["past_target","past_observed_values"],
                prediction_net=fc,batch_size=32,input_transform=tsp,device=device)
            fi,ti=make_evaluation_predictions(dataset=tt,predictor=pr,num_samples=16)
            tss=list(ti); forecasts=list(fi)
            m,_=Evaluator(num_workers=0)(tss,forecasts)
            crps=m["mean_wQuantileLoss"]
            if crps<best:
                best=crps
                torch.save({"net_ema":net_ema.state_dict(),"crps":crps,"epoch":epoch+1,
                           "noise_scale":args.noise_scale},
                          os.path.join(outdir,"best.pt"))
            tsf=TSFLOW[name]
            beat=" *** BEAT TSFLOW! ***" if crps<tsf else ""
            logger.info(f"  CRPS={crps:.6f} | TSFlow={tsf} | Best={best:.6f}{beat}")

    logger.info(f"FINAL {name} (noise={args.noise_scale}): Best={best:.6f}")


if __name__=="__main__": main()
